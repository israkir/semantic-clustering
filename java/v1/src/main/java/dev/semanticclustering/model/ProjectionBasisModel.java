package dev.semanticclustering.model;

import java.util.Arrays;

public record ProjectionBasisModel(double[] origin, double[] axisX, double[] axisY) {
    public ProjectionBasisModel {
        origin = Arrays.copyOf(origin, origin.length);
        axisX = Arrays.copyOf(axisX, axisX.length);
        axisY = Arrays.copyOf(axisY, axisY.length);
    }
}
