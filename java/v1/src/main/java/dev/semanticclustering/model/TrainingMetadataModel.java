package dev.semanticclustering.model;

public record TrainingMetadataModel(
        String embeddingModel,
        int embeddingDimensions,
        boolean normalizeEmbeddings,
        int trainingSize,
        long processingDurationMs) {
}
