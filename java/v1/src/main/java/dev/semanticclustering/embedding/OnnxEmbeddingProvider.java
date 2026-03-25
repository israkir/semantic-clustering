package dev.semanticclustering.embedding;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import dev.semanticclustering.util.VectorMath;

import java.io.IOException;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.locks.ReentrantLock;

/**
 * ONNX Runtime inference for sentence embeddings (pooling or pooled output head).
 */
public final class OnnxEmbeddingProvider implements EmbeddingProvider, AutoCloseable {
    private final OrtEnvironment environment;
    private final OrtSession session;
    private final BgeWordPieceTokenizer tokenizer;
    private final ReentrantLock inferenceLock = new ReentrantLock();
    private final String modelName;

    public OnnxEmbeddingProvider(Path onnxModelPath, Path vocabPath, int maxSequenceLength, String modelName)
            throws OrtException, IOException {
        this.environment = OrtEnvironment.getEnvironment();
        this.session = environment.createSession(onnxModelPath.toString(), new OrtSession.SessionOptions());
        this.tokenizer = new BgeWordPieceTokenizer(vocabPath, maxSequenceLength);
        this.modelName = modelName;
        validateModelContract(maxSequenceLength);
    }

    @Override
    public List<double[]> embed(List<String> texts) {
        List<double[]> embeddings = new ArrayList<>(texts.size());
        for (String text : texts) {
            float[] raw = embedSingle(text);
            double[] vector = new double[raw.length];
            for (int i = 0; i < raw.length; i++) {
                vector[i] = raw[i];
            }
            embeddings.add(vector);
        }
        return embeddings;
    }

    @Override
    public String modelName() {
        return modelName;
    }

    private float[] embedSingle(String text) {
        TokenizedInput encoded = tokenizer.encode(text);
        inferenceLock.lock();
        try {
            return runInference(encoded);
        } finally {
            inferenceLock.unlock();
        }
    }

    private float[] runInference(TokenizedInput encoded) {
        try {
            long[][] inputIds = new long[][]{encoded.inputIds()};
            long[][] attentionMask = new long[][]{encoded.attentionMask()};
            long[][] tokenTypeIds = new long[][]{encoded.tokenTypeIds()};

            try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(inputIds[0]), new long[]{1, inputIds[0].length});
                 OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(attentionMask[0]), new long[]{1, attentionMask[0].length});
                 OnnxTensor tokenTypeTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(tokenTypeIds[0]), new long[]{1, tokenTypeIds[0].length})) {
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("input_ids", inputIdsTensor);
                inputs.put("attention_mask", attentionMaskTensor);
                if (session.getInputNames().contains("token_type_ids")) {
                    inputs.put("token_type_ids", tokenTypeTensor);
                }

                try (OrtSession.Result result = session.run(inputs)) {
                    return VectorMath.l2Normalize(extractEmbedding(result, encoded.attentionMask()));
                }
            }
        } catch (OrtException e) {
            throw new IllegalStateException("ONNX inference failed", e);
        }
    }

    private float[] extractEmbedding(OrtSession.Result result, long[] attentionMask) throws OrtException {
        if (result.get("sentence_embedding").isPresent()) {
            Object value = result.get("sentence_embedding").get().getValue();
            if (value instanceof float[][] pooled && pooled.length > 0) {
                return pooled[0];
            }
        }
        Object output = result.get(0).getValue();
        if (output instanceof float[][][] hiddenStates) {
            return meanPool(hiddenStates, attentionMask);
        }
        if (output instanceof float[][] pooled && pooled.length > 0) {
            return pooled[0];
        }
        throw new IllegalStateException("Unexpected ONNX output type");
    }

    private float[] meanPool(float[][][] hiddenStates, long[] attentionMask) {
        float[][] sequenceEmbeddings = hiddenStates[0];
        int hiddenSize = sequenceEmbeddings[0].length;
        float[] pooled = new float[hiddenSize];
        float tokenCount = 0f;
        for (int tokenIndex = 0; tokenIndex < sequenceEmbeddings.length && tokenIndex < attentionMask.length; tokenIndex++) {
            if (attentionMask[tokenIndex] == 0) {
                continue;
            }
            tokenCount += 1f;
            for (int dim = 0; dim < hiddenSize; dim++) {
                pooled[dim] += sequenceEmbeddings[tokenIndex][dim];
            }
        }
        if (tokenCount > 0f) {
            for (int dim = 0; dim < hiddenSize; dim++) {
                pooled[dim] /= tokenCount;
            }
        }
        return pooled;
    }

    private void validateModelContract(int maxSequenceLength) throws OrtException {
        Set<String> inputNames = session.getInputNames();
        if (!inputNames.contains("input_ids") || !inputNames.contains("attention_mask")) {
            throw new IllegalArgumentException("Model must expose input_ids and attention_mask inputs");
        }
        Map<String, NodeInfo> inputInfo = session.getInputInfo();
        validateInputShape(inputInfo, "input_ids", maxSequenceLength);
        validateInputShape(inputInfo, "attention_mask", maxSequenceLength);
        if (inputInfo.containsKey("token_type_ids")) {
            validateInputShape(inputInfo, "token_type_ids", maxSequenceLength);
        }
    }

    private void validateInputShape(Map<String, NodeInfo> inputInfo, String inputName, int maxSequenceLength) {
        NodeInfo nodeInfo = inputInfo.get(inputName);
        if (!(nodeInfo.getInfo() instanceof TensorInfo tensorInfo)) {
            throw new IllegalArgumentException("Input '" + inputName + "' missing tensor metadata");
        }
        long[] shape = tensorInfo.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("Input '" + inputName + "' must be rank-2");
        }
        long sequenceLength = shape[1];
        if (sequenceLength > 0 && sequenceLength != maxSequenceLength) {
            throw new IllegalArgumentException("Input '" + inputName + "' sequence length mismatch");
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
