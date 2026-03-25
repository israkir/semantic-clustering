package dev.semanticclustering.processing;

import smile.manifold.UMAP;

/**
 * Smile UMAP 2D projection for visualization.
 */
final class UmapProjection {

    private static final int DEFAULT_N_NEIGHBORS = 4;
    private static final int TARGET_DIMENSIONS = 2;
    private static final int EPOCHS = 100;

    private UmapProjection() {
    }

    static double[][] project(double[][] vectors) {
        if (vectors == null || vectors.length == 0) {
            return new double[0][];
        }
        var options = new UMAP.Options(
                DEFAULT_N_NEIGHBORS,
                TARGET_DIMENSIONS,
                EPOCHS,
                1.0,
                0.1,
                1.0,
                5,
                1.0,
                1.0);
        return UMAP.fit(vectors, options);
    }
}
