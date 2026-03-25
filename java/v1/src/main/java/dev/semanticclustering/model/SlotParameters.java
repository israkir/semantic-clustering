package dev.semanticclustering.model;

public record SlotParameters(
        MetricKind metric,
        boolean normalizeEmbeddings,
        int minClusterSize,
        int neighborCount,
        int numThreads,
        NeighborQueryStrategyKind neighborQueryStrategy,
        String projectionMethod) {
}
