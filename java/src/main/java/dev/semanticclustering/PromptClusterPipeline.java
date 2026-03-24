package dev.semanticclustering;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import dev.semanticclustering.config.TextNormalizeConfig;
import dev.semanticclustering.embedding.OnnxEmbeddingProvider;
import dev.semanticclustering.model.ClusterAssignment;
import dev.semanticclustering.model.MetricKind;
import dev.semanticclustering.model.NeighborQueryStrategyKind;
import dev.semanticclustering.model.ProcessingResult;
import dev.semanticclustering.model.PromptRecord;
import dev.semanticclustering.model.SlotParameters;
import dev.semanticclustering.processing.ClusteringPipeline;
import dev.semanticclustering.processing.HdbscanClusterer;
import dev.semanticclustering.processing.ModelSerializationAdapter;
import dev.semanticclustering.processing.PreprocessingPipeline;
import dev.semanticclustering.processing.QualityMetricsCalculator;
import dev.semanticclustering.processing.TextNormalizer;
import dev.semanticclustering.processing.TribuoDatasetFactory;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.awt.Color;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.UUID;
import java.util.logging.Level;
import java.util.logging.Logger;

/** CLI: BGE-small ONNX → Tribuo HDBSCAN* → Smile UMAP (PCA fallback), matching tool-mismatch defaults. */
public final class PromptClusterPipeline {

    private static final String TAG = "[JAVA|pipeline]";
    private static final int BANNER_WIDTH = 76;
    private static final String BGE_ONNX_SUBDIR = "model/onnx/bge-small-en-v1.5";

    static {
        Logger.getLogger("org.tribuo.clustering.hdbscan.HdbscanTrainer").setLevel(Level.WARNING);
    }

    private PromptClusterPipeline() {
    }

    private static void log(String msg) {
        System.out.println(TAG + " " + msg);
    }

    public static void main(String[] args) throws Exception {
        Path repoRoot = inferRepoRoot();
        Path promptsPath;
        Path outPath;
        Path onnxPath;
        Path vocabPath;
        int maxSeqLen = 512;
        String modelLabel = "bge-small-en-v1.5";

        if (args.length >= 4) {
            promptsPath = Path.of(args[0]);
            outPath = Path.of(args[1]);
            onnxPath = Path.of(args[2]);
            vocabPath = Path.of(args[3]);
        } else if (args.length >= 2) {
            promptsPath = Path.of(args[0]);
            outPath = Path.of(args[1]);
            Path[] resolved = resolveOnnxPaths(repoRoot);
            onnxPath = resolved[0];
            vocabPath = resolved[1];
        } else if (args.length == 1) {
            promptsPath = repoRoot.resolve("data/prompts.txt");
            outPath = Path.of(args[0]);
            Path[] resolved = resolveOnnxPaths(repoRoot);
            onnxPath = resolved[0];
            vocabPath = resolved[1];
        } else {
            promptsPath = repoRoot.resolve("data/prompts.txt");
            outPath = repoRoot.resolve("outputs/java_tribuo_hdbscan.png");
            Path[] resolved = resolveOnnxPaths(repoRoot);
            onnxPath = resolved[0];
            vocabPath = resolved[1];
        }

        String line = "=".repeat(BANNER_WIDTH);
        System.out.println();
        System.out.println(line);
        System.out.println("  BGE-small ONNX · Tribuo HDBSCAN* · UMAP (PCA fallback) — tool-mismatch defaults");
        System.out.println("  Noise: cluster id 0");
        System.out.println(line);

        if (Files.notExists(onnxPath) || Files.notExists(vocabPath)) {
            System.err.println(TAG + " ONNX model or vocab not found.");
            System.err.println("  Tried: " + onnxPath.toAbsolutePath());
            System.err.println("  Tried: " + vocabPath.toAbsolutePath());
            System.err.println("  Expected under " + repoRoot.resolve(BGE_ONNX_SUBDIR).toAbsolutePath());
            System.err.println("  After clone: git lfs install && git lfs pull  (or: make git-lfs-pull)");
            System.err.println("  Or set SEMANTIC_CLUSTERING_ONNX_MODEL and SEMANTIC_CLUSTERING_ONNX_VOCAB");
            System.err.println("  Usage: java … PromptClusterPipeline <prompts.txt> <out.png> [model.onnx] [vocab.txt]");
            System.exit(2);
        }
        log("ONNX: " + onnxPath.toAbsolutePath());

        log("Prompts file: " + promptsPath.toAbsolutePath());
        List<PromptRecord> prompts = loadPromptRecords(promptsPath);
        log("Loaded " + prompts.size() + " prompt(s)");
        Files.createDirectories(outPath.getParent());

        try (OnnxEmbeddingProvider embeddingProvider = new OnnxEmbeddingProvider(onnxPath, vocabPath, maxSeqLen, modelLabel)) {
            TextNormalizer normalizer = new TextNormalizer(TextNormalizeConfig.defaults());
            PreprocessingPipeline preprocessing = new PreprocessingPipeline(normalizer, embeddingProvider);
            ModelSerializationAdapter serialization = new ModelSerializationAdapter();
            TribuoDatasetFactory datasetFactory = new TribuoDatasetFactory(serialization);
            HdbscanClusterer clusterer = new HdbscanClusterer(datasetFactory, serialization);
            ClusteringPipeline pipeline = new ClusteringPipeline(preprocessing, clusterer, new QualityMetricsCalculator());

            // Defaults aligned with tool-mismatch-clustering application.yaml + service resolution:
            // cosine, normalize embeddings, min_cluster_size 5, neighbor_count 0 → min_cluster_size, brute_force, umap.
            int minCluster = 5;
            int threads = Math.max(1, Math.min(8, Runtime.getRuntime().availableProcessors()));
            SlotParameters slotParameters = new SlotParameters(
                    MetricKind.COSINE,
                    true,
                    minCluster,
                    minCluster,
                    threads,
                    NeighborQueryStrategyKind.BRUTE_FORCE,
                    "umap");

            ProcessingResult result = pipeline.process(prompts, slotParameters);

            saveChart(result, outPath);
            writeJson(promptsPath, outPath, result);
            log("Wrote image  " + outPath.toAbsolutePath());
            log("Wrote JSON   " + siblingJson(outPath).toAbsolutePath());
            log("Done.");
        }
    }

    /**
     * When {@code mvn exec:java} runs from {@code java/}, {@code user.dir} is that module; when running the
     * JAR from the repo root, {@code user.dir} is the repo. Prefer parent of {@code java/} when that looks
     * like this project.
     */
    private static Path inferRepoRoot() {
        Path cwd = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
        if (Files.isRegularFile(cwd.resolve("pom.xml"))
                && "java".equalsIgnoreCase(String.valueOf(cwd.getFileName()))) {
            Path parent = cwd.getParent();
            if (parent != null) {
                return parent;
            }
        }
        return cwd;
    }

    /** Resolves {@code model.onnx} and {@code vocab.txt}: env vars, then {@code model/onnx/bge-small-en-v1.5/}. */
    private static Path[] resolveOnnxPaths(Path repoRoot) {
        String envOnnx = System.getenv("SEMANTIC_CLUSTERING_ONNX_MODEL");
        String envVocab = System.getenv("SEMANTIC_CLUSTERING_ONNX_VOCAB");
        if (envOnnx != null && !envOnnx.isBlank() && envVocab != null && !envVocab.isBlank()) {
            return new Path[] { Path.of(envOnnx), Path.of(envVocab) };
        }
        Path base = repoRoot.resolve(BGE_ONNX_SUBDIR);
        return new Path[] { base.resolve("model.onnx"), base.resolve("vocab.txt") };
    }

    private static List<PromptRecord> loadPromptRecords(Path path) throws IOException {
        if (!Files.isRegularFile(path)) {
            throw new IOException("Prompts file not found: " + path.toAbsolutePath());
        }
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        List<PromptRecord> records = new ArrayList<>();
        int order = 0;
        for (String line : lines) {
            String s = line.strip();
            if (s.isEmpty() || s.startsWith("#")) {
                continue;
            }
            records.add(new PromptRecord(UUID.randomUUID().toString(), s, 0, order++));
        }
        if (records.isEmpty()) {
            throw new IOException("No prompts in file: " + path.toAbsolutePath());
        }
        return records;
    }

    private static void saveChart(ProcessingResult result, Path outPath) throws IOException {
        Map<Integer, List<double[]>> byLabel = new LinkedHashMap<>();
        for (ClusterAssignment pt : result.dataPoints()) {
            byLabel.computeIfAbsent(pt.clusterId(), k -> new ArrayList<>())
                    .add(new double[] { pt.plotPoint().x(), pt.plotPoint().y() });
        }

        XYChart chart = new XYChartBuilder()
                .width(960)
                .height(640)
                .title("Prompt Intent Clustering")
                .xAxisTitle("X")
                .yAxisTitle("Y")
                .build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        chart.getStyler().setLegendVisible(true);
        chart.getStyler().setMarkerSize(10);

        Queue<Color> colors = new ArrayDeque<>(List.of(
                Color.CYAN, Color.GREEN, Color.MAGENTA, Color.ORANGE, Color.BLUE,
                new Color(0.4f, 0.7f, 0.2f), new Color(0.6f, 0.2f, 0.8f)));

        List<Integer> keys = new ArrayList<>(byLabel.keySet());
        keys.sort(Comparator.naturalOrder());
        for (Integer clusterId : keys) {
            Color color = clusterId <= 0 ? Color.RED : colors.poll();
            if (color == null) {
                color = Color.DARK_GRAY;
            }
            String seriesName = clusterId <= 0 ? "noise (" + clusterId + ")" : "cluster " + clusterId;
            List<double[]> pts = byLabel.get(clusterId);
            double[] xs = new double[pts.size()];
            double[] ys = new double[pts.size()];
            for (int i = 0; i < pts.size(); i++) {
                xs[i] = pts.get(i)[0];
                ys[i] = pts.get(i)[1];
            }
            var series = chart.addSeries(seriesName, xs, ys);
            series.setMarkerColor(color);
            series.setMarker(SeriesMarkers.CIRCLE);
        }

        BitmapEncoder.saveBitmap(chart, outPath.toAbsolutePath().toString(), BitmapEncoder.BitmapFormat.PNG);
    }

    private static void writeJson(Path promptsPath, Path imageOut, ProcessingResult result) throws IOException {
        Map<Integer, List<String>> buckets = new LinkedHashMap<>();
        for (ClusterAssignment pt : result.dataPoints()) {
            String text = pt.normalizedText() != null ? pt.normalizedText() : pt.rawText();
            buckets.computeIfAbsent(pt.clusterId(), k -> new ArrayList<>()).add(text);
        }
        List<Integer> order = new ArrayList<>(buckets.keySet());
        order.sort(Comparator.naturalOrder());
        Map<String, List<String>> clusters = new LinkedHashMap<>();
        for (int lab : order) {
            clusters.put(clusterJsonKey(lab), buckets.get(lab));
        }
        Path jsonOut = siblingJson(imageOut);
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("program", "semantic-clustering");
        root.put("prompts_file", promptsPath.toAbsolutePath().toString());
        root.put("output_image", imageOut.toAbsolutePath().toString());
        root.put("output_json", jsonOut.toAbsolutePath().toString());
        root.put("clusters", clusters);
        root.put("metadata", result.metadata());
        Gson gson = new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create();
        Files.writeString(jsonOut, gson.toJson(root), StandardCharsets.UTF_8);
    }

    private static String clusterJsonKey(int tribuoLabel) {
        if (tribuoLabel <= 0) {
            return "noise";
        }
        return "cluster_" + tribuoLabel;
    }

    private static Path siblingJson(Path imagePath) {
        String fn = imagePath.getFileName().toString();
        String base = fn.endsWith(".png") ? fn.substring(0, fn.length() - 4) : fn;
        return imagePath.resolveSibling(base + ".json");
    }
}
