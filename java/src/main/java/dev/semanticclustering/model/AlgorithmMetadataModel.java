package dev.semanticclustering.model;

public record AlgorithmMetadataModel(
        String algorithm,
        String library,
        String version,
        MetricKind distanceMetric,
        int minClusterSize,
        int neighborCount,
        int numThreads,
        NeighborQueryStrategyKind neighborQueryStrategy) {
}
