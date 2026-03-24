package dev.semanticclustering.model;

import java.util.Arrays;
import java.util.List;

public record ClusterDefinitionModel(
        byte[] serializedModel,
        AlgorithmMetadataModel algorithmMetadata,
        TrainingMetadataModel trainingMetadata,
        List<ClusterSummaryModel> clusterSummaries,
        ProjectionBasisModel projection) {
    public ClusterDefinitionModel {
        serializedModel = Arrays.copyOf(serializedModel, serializedModel.length);
        clusterSummaries = List.copyOf(clusterSummaries);
    }

    @Override
    public byte[] serializedModel() {
        return Arrays.copyOf(serializedModel, serializedModel.length);
    }
}
