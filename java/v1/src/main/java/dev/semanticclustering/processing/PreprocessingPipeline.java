package dev.semanticclustering.processing;

import dev.semanticclustering.embedding.EmbeddingProvider;
import dev.semanticclustering.model.PromptRecord;
import dev.semanticclustering.util.VectorMath;

import java.util.ArrayList;
import java.util.List;

/**
 * Text normalization plus embedding provider.
 */
public final class PreprocessingPipeline {
    private final TextNormalizer textNormalizer;
    private final EmbeddingProvider embeddingProvider;

    public PreprocessingPipeline(TextNormalizer textNormalizer, EmbeddingProvider embeddingProvider) {
        this.textNormalizer = textNormalizer;
        this.embeddingProvider = embeddingProvider;
    }

    public double[][] process(List<PromptRecord> prompts, boolean normalizeEmbeddings) {
        if (prompts.isEmpty()) {
            return new double[0][];
        }

        List<String> normalizedTexts = new ArrayList<>(prompts.size());
        for (PromptRecord prompt : prompts) {
            String normalized = textNormalizer.normalize(prompt.rawText());
            prompt.normalizedText(normalized);
            normalizedTexts.add(normalized);
        }

        List<double[]> rawEmbeddings = embeddingProvider.embed(normalizedTexts);
        double[][] finalVectors = new double[rawEmbeddings.size()][];

        for (int i = 0; i < rawEmbeddings.size(); i++) {
            double[] raw = rawEmbeddings.get(i);
            double[] processed = normalizeEmbeddings ? VectorMath.l2Normalize(raw) : raw.clone();
            prompts.get(i).embedding(processed);
            finalVectors[i] = processed;
        }

        return finalVectors;
    }

    public String modelName() {
        return embeddingProvider.modelName();
    }
}
