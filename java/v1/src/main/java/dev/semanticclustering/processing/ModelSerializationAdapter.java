package dev.semanticclustering.processing;

import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.Model;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.hdbscan.HdbscanModel;
import org.tribuo.protos.core.ModelProto;

/**
 * Tribuo HDBSCAN model protobuf serialize/deserialize.
 */
public final class ModelSerializationAdapter {

    public byte[] serialize(HdbscanModel model) {
        return model.serialize().toByteArray();
    }

    public HdbscanModel deserialize(byte[] payload) {
        Model<?> model;
        try {
            model = Model.deserialize(ModelProto.parseFrom(payload));
        } catch (InvalidProtocolBufferException e) {
            throw new IllegalArgumentException("invalid Tribuo model bytes", e);
        }
        if (!(model instanceof HdbscanModel hdbscanModel)) {
            throw new IllegalArgumentException("serialized model is not a Tribuo HDBSCAN model");
        }
        return hdbscanModel;
    }

    public ClusterID unassignedClusterId() {
        return org.tribuo.clustering.ClusteringFactory.UNASSIGNED_CLUSTER_ID;
    }
}
