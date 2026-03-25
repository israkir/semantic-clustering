package dev.semanticclustering.processing;

import dev.semanticclustering.model.ClusterAssignment;
import dev.semanticclustering.model.ClusterDefinitionModel;
import dev.semanticclustering.model.ClusterSummaryModel;
import dev.semanticclustering.model.PlotPointModel;
import dev.semanticclustering.model.ProcessingResult;
import dev.semanticclustering.model.PromptRecord;
import dev.semanticclustering.model.QualityMetricsModel;
import dev.semanticclustering.model.SlotParameters;
import dev.semanticclustering.model.TrainingMetadataModel;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Embedding → Tribuo HDBSCAN → summaries, metrics, and 2D plot coordinates.
 */
public final class ClusteringPipeline {
    private final PreprocessingPipeline preprocessingPipeline;
    private final HdbscanClusterer hdbscanClusterer;
    private final QualityMetricsCalculator metricsCalculator;

    public ClusteringPipeline(
            PreprocessingPipeline preprocessingPipeline,
            HdbscanClusterer hdbscanClusterer,
            QualityMetricsCalculator metricsCalculator) {
        this.preprocessingPipeline = preprocessingPipeline;
        this.hdbscanClusterer = hdbscanClusterer;
        this.metricsCalculator = metricsCalculator;
    }

    public ProcessingResult process(List<PromptRecord> prompts, SlotParameters parameters) {
        Instant startedAt = Instant.now();

        double[][] vectors = preprocessingPipeline.process(prompts, parameters.normalizeEmbeddings());

        HdbscanClusterer.ClusterOutcome outcome = hdbscanClusterer.cluster(vectors, parameters);
        int[] labels = outcome.labels();
        long processingDurationMs = Duration.between(startedAt, Instant.now()).toMillis();

        Map<Integer, List<String>> clusterIds = new LinkedHashMap<>();
        Map<Integer, List<Integer>> clusterIndexes = new LinkedHashMap<>();
        for (int i = 0; i < prompts.size(); i++) {
            if (labels[i] > 0) {
                clusterIds.computeIfAbsent(labels[i], ignored -> new ArrayList<>()).add(prompts.get(i).promptId());
                clusterIndexes.computeIfAbsent(labels[i], ignored -> new ArrayList<>()).add(i);
            }
        }

        List<double[]> clusteredVectors = new ArrayList<>();
        for (List<Integer> indexes : clusterIndexes.values()) {
            for (Integer index : indexes) {
                clusteredVectors.add(vectors[index]);
            }
        }
        double[][] projectionVectors = clusteredVectors.isEmpty() ? vectors : clusteredVectors.toArray(double[][]::new);
        ProjectionSpace definitionProjectionSpace = ProjectionSpace.fit(projectionVectors);

        List<ClusterSummaryModel> clusters = new ArrayList<>(clusterIds.size());
        for (Map.Entry<Integer, List<String>> entry : clusterIds.entrySet()) {
            clusters.add(new ClusterSummaryModel(entry.getKey(), List.copyOf(entry.getValue())));
        }
        clusters.sort(Comparator.comparingInt(ClusterSummaryModel::clusterId));

        ClusterDefinitionModel clusterDefinition = buildClusterDefinition(
                vectors,
                processingDurationMs,
                parameters.normalizeEmbeddings(),
                outcome,
                clusters,
                definitionProjectionSpace);

        boolean useUmap = "umap".equalsIgnoreCase(parameters.projectionMethod());
        double[][] plotCoords;
        if (useUmap && vectors.length > 0) {
            plotCoords = runUmapWithTimeout(vectors, clusterDefinition);
        } else {
            plotCoords = fallbackToPca(vectors, clusterDefinition);
        }

        List<ClusterAssignment> dataPoints = new ArrayList<>(prompts.size());
        List<ClusterAssignment> noisePoints = new ArrayList<>();
        for (int i = 0; i < prompts.size(); i++) {
            PromptRecord prompt = prompts.get(i);
            PlotPointModel plotPoint = plotCoords.length > i
                    ? new PlotPointModel(plotCoords[i][0], plotCoords[i][1])
                    : new PlotPointModel(0.0, 0.0);
            ClusterAssignment assignment = new ClusterAssignment(
                    prompt.promptId(),
                    prompt.rawText(),
                    prompt.normalizedText(),
                    labels[i],
                    plotPoint,
                    outcome.outlierScores().length > i ? outcome.outlierScores()[i] : null);
            dataPoints.add(assignment);
            if (labels[i] <= 0) {
                noisePoints.add(assignment);
            }
        }

        QualityMetricsModel qualityMetrics = metricsCalculator.calculate(labels);

        Map<String, String> metadata = new LinkedHashMap<>();
        metadata.put("algorithm", outcome.algorithmMetadata().algorithm().toLowerCase());
        metadata.put("library", outcome.algorithmMetadata().library().toLowerCase());
        metadata.put("tribuo_version", outcome.algorithmMetadata().version());
        metadata.put("metric", parameters.metric().name().toLowerCase());
        metadata.put("min_cluster_size", Integer.toString(parameters.minClusterSize()));
        metadata.put("neighbor_count", Integer.toString(parameters.neighborCount()));
        metadata.put("num_threads", Integer.toString(parameters.numThreads()));
        metadata.put("neighbor_query_strategy", parameters.neighborQueryStrategy().name().toLowerCase());
        metadata.put("normalize_embeddings", Boolean.toString(parameters.normalizeEmbeddings()));
        metadata.put("embedding_model", preprocessingPipeline.modelName());
        metadata.put("embedding_dimensions", vectors.length == 0 ? "0" : Integer.toString(vectors[0].length));
        metadata.put("projection_method", parameters.projectionMethod());
        metadata.put("processing_duration_ms", Long.toString(processingDurationMs));

        return new ProcessingResult(clusters, dataPoints, noisePoints, qualityMetrics, clusterDefinition, Map.copyOf(metadata));
    }

    private static final long UMAP_TIMEOUT_SECONDS = 180;

    private double[][] runUmapWithTimeout(double[][] vectors, ClusterDefinitionModel clusterDefinition) {
        ExecutorService executor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "umap-worker");
            t.setDaemon(true);
            return t;
        });
        try {
            Future<double[][]> future = executor.submit(() -> UmapProjection.project(vectors));
            double[][] result = future.get(UMAP_TIMEOUT_SECONDS, TimeUnit.SECONDS);
            return result != null ? result : fallbackToPca(vectors, clusterDefinition);
        } catch (Exception e) {
            return fallbackToPca(vectors, clusterDefinition);
        } finally {
            executor.shutdownNow();
        }
    }

    private double[][] fallbackToPca(double[][] vectors, ClusterDefinitionModel clusterDefinition) {
        ProjectionSpace projectionSpace = ProjectionSpace.from(clusterDefinition.projection());
        double[][] plotCoords = new double[vectors.length][2];
        for (int i = 0; i < vectors.length; i++) {
            PlotPointModel p = projectionSpace.project(vectors[i]);
            plotCoords[i] = new double[] { p.x(), p.y() };
        }
        return plotCoords;
    }

    private ClusterDefinitionModel buildClusterDefinition(
            double[][] vectors,
            long processingDurationMs,
            boolean normalizeEmbeddings,
            HdbscanClusterer.ClusterOutcome outcome,
            List<ClusterSummaryModel> clusters,
            ProjectionSpace projectionSpace) {
        return new ClusterDefinitionModel(
                outcome.serializedModel(),
                outcome.algorithmMetadata(),
                new TrainingMetadataModel(
                        preprocessingPipeline.modelName(),
                        vectors.length == 0 ? 0 : vectors[0].length,
                        normalizeEmbeddings,
                        vectors.length,
                        processingDurationMs),
                clusters,
                projectionSpace.basis());
    }
}
