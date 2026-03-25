package dev.semanticclustering.embedding;

import java.util.List;

public interface EmbeddingProvider {
    List<double[]> embed(List<String> texts);

    String modelName();
}
