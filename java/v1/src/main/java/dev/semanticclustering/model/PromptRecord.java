package dev.semanticclustering.model;

public final class PromptRecord {
    private final String promptId;
    private final String rawText;
    private final int uploadBatchIndex;
    private final int insertionOrder;
    private String normalizedText;
    private double[] embedding;

    public PromptRecord(String promptId, String rawText, int uploadBatchIndex, int insertionOrder) {
        this.promptId = promptId;
        this.rawText = rawText;
        this.uploadBatchIndex = uploadBatchIndex;
        this.insertionOrder = insertionOrder;
    }

    public String promptId() {
        return promptId;
    }

    public String rawText() {
        return rawText;
    }

    public int uploadBatchIndex() {
        return uploadBatchIndex;
    }

    public int insertionOrder() {
        return insertionOrder;
    }

    public String normalizedText() {
        return normalizedText;
    }

    public void normalizedText(String normalizedText) {
        this.normalizedText = normalizedText;
    }

    public double[] embedding() {
        return embedding == null ? null : embedding.clone();
    }

    public void embedding(double[] embedding) {
        this.embedding = embedding == null ? null : embedding.clone();
    }
}
