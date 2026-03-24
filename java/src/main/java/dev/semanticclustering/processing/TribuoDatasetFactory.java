package dev.semanticclustering.processing;

import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ClusteringFactory;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.SimpleDataSourceProvenance;

import java.time.OffsetDateTime;
import java.util.Arrays;

/**
 * Builds Tribuo datasets from dense embedding rows.
 */
public final class TribuoDatasetFactory {
    private static final String SOURCE_NAME = "semantic-clustering";

    private final ClusteringFactory clusteringFactory = new ClusteringFactory();
    private final ModelSerializationAdapter serializationAdapter;

    public TribuoDatasetFactory(ModelSerializationAdapter serializationAdapter) {
        this.serializationAdapter = serializationAdapter;
    }

    public MutableDataset<ClusterID> createDataset(double[][] vectors) {
        MutableDataset<ClusterID> dataset = new MutableDataset<>(
                new SimpleDataSourceProvenance(SOURCE_NAME, OffsetDateTime.now(), clusteringFactory),
                clusteringFactory);
        for (double[] vector : vectors) {
            dataset.add(createExample(vector));
        }
        return dataset;
    }

    public Example<ClusterID> createExample(double[] vector) {
        return new ArrayExample<>(
                serializationAdapter.unassignedClusterId(),
                featureNames(vector.length),
                Arrays.copyOf(vector, vector.length));
    }

    private String[] featureNames(int dimensions) {
        String[] names = new String[dimensions];
        for (int i = 0; i < dimensions; i++) {
            names[i] = "embedding_" + i;
        }
        return names;
    }
}
