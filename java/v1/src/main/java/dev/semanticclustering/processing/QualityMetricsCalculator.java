package dev.semanticclustering.processing;

import dev.semanticclustering.model.QualityMetricsModel;

import java.util.HashSet;
import java.util.OptionalDouble;
import java.util.Set;

/**
 * Label-level clustering summary metrics.
 */
public final class QualityMetricsCalculator {

    public QualityMetricsModel calculate(int[] labels) {
        int totalPoints = labels.length;
        int noisePoints = 0;
        Set<Integer> clusterIds = new HashSet<>();
        for (int label : labels) {
            if (label <= 0) {
                noisePoints += 1;
            } else {
                clusterIds.add(label);
            }
        }
        int clusteredPoints = totalPoints - noisePoints;
        int clusterCount = clusterIds.size();
        double noiseRatio = totalPoints == 0 ? 0.0 : (double) noisePoints / totalPoints;
        double clusteredRatio = totalPoints == 0 ? 0.0 : (double) clusteredPoints / totalPoints;
        return new QualityMetricsModel(
                totalPoints,
                clusteredPoints,
                noisePoints,
                clusterCount,
                noiseRatio,
                clusteredRatio,
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalDouble.empty());
    }
}
