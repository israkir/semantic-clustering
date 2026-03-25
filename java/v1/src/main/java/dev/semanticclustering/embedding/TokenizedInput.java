package dev.semanticclustering.embedding;

public record TokenizedInput(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) {
    public TokenizedInput {
        inputIds = inputIds.clone();
        attentionMask = attentionMask.clone();
        tokenTypeIds = tokenTypeIds.clone();
    }
}
