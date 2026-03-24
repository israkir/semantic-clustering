package dev.semanticclustering.model;

import java.util.List;

public record ClusterSummaryModel(int clusterId, List<String> promptIds) {
    public ClusterSummaryModel {
        promptIds = List.copyOf(promptIds);
    }

    public int size() {
        return promptIds.size();
    }
}
