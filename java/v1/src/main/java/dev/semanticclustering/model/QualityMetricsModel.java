package dev.semanticclustering.model;

import java.util.OptionalDouble;

public record QualityMetricsModel(
        int totalPoints,
        int clusteredPoints,
        int noisePoints,
        int clusterCount,
        double noiseRatio,
        double clusteredRatio,
        OptionalDouble silhouetteScore,
        OptionalDouble daviesBouldinIndex,
        OptionalDouble calinskiHarabaszScore) {
}
