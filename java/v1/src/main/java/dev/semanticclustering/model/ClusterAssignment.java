package dev.semanticclustering.model;

public record ClusterAssignment(
        String promptId,
        String rawText,
        String normalizedText,
        int clusterId,
        PlotPointModel plotPoint,
        Double outlierScore) {
}
