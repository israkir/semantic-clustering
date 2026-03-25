package dev.semanticclustering.processing;

import dev.semanticclustering.model.AlgorithmMetadataModel;
import dev.semanticclustering.model.MetricKind;
import dev.semanticclustering.model.NeighborQueryStrategyKind;
import dev.semanticclustering.model.SlotParameters;
import org.tribuo.Tribuo;
import org.tribuo.clustering.hdbscan.HdbscanModel;
import org.tribuo.clustering.hdbscan.HdbscanTrainer;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;

import java.util.List;

/**
 * Tribuo HDBSCAN training and model serialization.
 */
public final class HdbscanClusterer {
    private final TribuoDatasetFactory datasetFactory;
    private final ModelSerializationAdapter serializationAdapter;

    public HdbscanClusterer(TribuoDatasetFactory datasetFactory, ModelSerializationAdapter serializationAdapter) {
        this.datasetFactory = datasetFactory;
        this.serializationAdapter = serializationAdapter;
    }

    public ClusterOutcome cluster(double[][] vectors, SlotParameters parameters) {
        if (vectors.length == 0) {
            return new ClusterOutcome(new int[0], new Double[0], new byte[0], algorithmMetadata(parameters));
        }

        HdbscanTrainer trainer = new HdbscanTrainer(
                parameters.minClusterSize(),
                distanceType(parameters.metric()).getDistance(),
                parameters.neighborCount(),
                parameters.numThreads(),
                queryFactoryType(parameters.neighborQueryStrategy()));

        HdbscanModel model = trainer.train(datasetFactory.createDataset(vectors));
        int[] labels = model.getClusterLabels().stream().mapToInt(Integer::intValue).toArray();
        List<Double> outlierScores = model.getOutlierScores();
        Double[] scores = outlierScores.toArray(Double[]::new);
        return new ClusterOutcome(labels, scores, serializationAdapter.serialize(model), algorithmMetadata(parameters));
    }

    private AlgorithmMetadataModel algorithmMetadata(SlotParameters parameters) {
        return new AlgorithmMetadataModel(
                "HDBSCAN",
                "Tribuo",
                Tribuo.VERSION,
                parameters.metric(),
                parameters.minClusterSize(),
                parameters.neighborCount(),
                parameters.numThreads(),
                parameters.neighborQueryStrategy());
    }

    private DistanceType distanceType(MetricKind metricKind) {
        return switch (metricKind) {
            case COSINE -> DistanceType.COSINE;
            case EUCLIDEAN -> DistanceType.L2;
        };
    }

    private NeighboursQueryFactoryType queryFactoryType(NeighborQueryStrategyKind strategyKind) {
        return switch (strategyKind) {
            case BRUTE_FORCE -> NeighboursQueryFactoryType.BRUTE_FORCE;
            case KD_TREE -> NeighboursQueryFactoryType.KD_TREE;
        };
    }

    public record ClusterOutcome(
            int[] labels,
            Double[] outlierScores,
            byte[] serializedModel,
            AlgorithmMetadataModel algorithmMetadata) {
    }
}
