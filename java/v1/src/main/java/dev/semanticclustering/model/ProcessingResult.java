package dev.semanticclustering.model;

import java.util.List;
import java.util.Map;

public record ProcessingResult(
        List<ClusterSummaryModel> clusters,
        List<ClusterAssignment> dataPoints,
        List<ClusterAssignment> noisePoints,
        QualityMetricsModel qualityMetrics,
        ClusterDefinitionModel clusterDefinition,
        Map<String, String> metadata) {
}
