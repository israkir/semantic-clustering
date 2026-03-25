package dev.semanticclustering.processing;

import dev.semanticclustering.model.PlotPointModel;
import dev.semanticclustering.model.ProjectionBasisModel;
import dev.semanticclustering.util.VectorMath;

import java.util.Arrays;

/**
 * 2D PCA projection for visualization and assignment plots.
 */
final class ProjectionSpace {
    private static final double EPSILON = 1.0e-12;

    private final ProjectionBasisModel basis;

    private ProjectionSpace(ProjectionBasisModel basis) {
        this.basis = basis;
    }

    static ProjectionSpace fit(double[][] vectors) {
        if (vectors.length == 0) {
            return new ProjectionSpace(new ProjectionBasisModel(new double[0], new double[0], new double[0]));
        }
        int dimensions = vectors[0].length;
        double[] origin = VectorMath.mean(vectors);
        if (dimensions == 0) {
            return new ProjectionSpace(new ProjectionBasisModel(origin, new double[0], new double[0]));
        }

        double[][] covariance = covariance(vectors, origin);
        double[] axisX = principalAxis(covariance);

        double[][] deflated = deflate(covariance, axisX);
        double[] axisY = principalAxis(deflated);

        if (norm(axisX) <= EPSILON) {
            axisX = canonicalAxis(dimensions, 0);
        }
        axisX = VectorMath.l2Normalize(axisX);

        axisY = orthogonalize(axisY, axisX);
        if (norm(axisY) <= EPSILON) {
            axisY = fallbackAxis(dimensions, axisX);
        }
        axisY = VectorMath.l2Normalize(axisY);

        return new ProjectionSpace(new ProjectionBasisModel(origin, axisX, axisY));
    }

    static ProjectionSpace from(ProjectionBasisModel basis) {
        return new ProjectionSpace(basis);
    }

    ProjectionBasisModel basis() {
        return basis;
    }

    PlotPointModel project(double[] vector) {
        if (basis.origin().length == 0) {
            return new PlotPointModel(0.0, 0.0);
        }
        double[] centered = VectorMath.subtract(vector, basis.origin());
        return new PlotPointModel(
                VectorMath.dot(centered, basis.axisX()),
                VectorMath.dot(centered, basis.axisY()));
    }

    private static double[][] covariance(double[][] vectors, double[] origin) {
        int dimensions = origin.length;
        double[][] covariance = new double[dimensions][dimensions];
        for (double[] vector : vectors) {
            double[] centered = VectorMath.subtract(vector, origin);
            for (int row = 0; row < dimensions; row++) {
                for (int column = 0; column < dimensions; column++) {
                    covariance[row][column] += centered[row] * centered[column];
                }
            }
        }
        double scale = vectors.length > 1 ? 1.0 / (vectors.length - 1) : 1.0;
        for (int row = 0; row < dimensions; row++) {
            for (int column = 0; column < dimensions; column++) {
                covariance[row][column] *= scale;
            }
        }
        return covariance;
    }

    private static double[] principalAxis(double[][] matrix) {
        int dimensions = matrix.length;
        if (dimensions == 0) {
            return new double[0];
        }
        double[] vector = new double[dimensions];
        Arrays.fill(vector, 1.0 / Math.sqrt(dimensions));
        for (int iteration = 0; iteration < 32; iteration++) {
            double[] next = multiply(matrix, vector);
            double n = norm(next);
            if (n <= EPSILON) {
                return new double[dimensions];
            }
            for (int i = 0; i < next.length; i++) {
                next[i] /= n;
            }
            vector = next;
        }
        return vector;
    }

    private static double[][] deflate(double[][] matrix, double[] axis) {
        int dimensions = matrix.length;
        double eigenvalue = VectorMath.dot(axis, multiply(matrix, axis));
        double[][] deflated = new double[dimensions][dimensions];
        for (int row = 0; row < dimensions; row++) {
            for (int column = 0; column < dimensions; column++) {
                deflated[row][column] = matrix[row][column] - eigenvalue * axis[row] * axis[column];
            }
        }
        return deflated;
    }

    private static double[] multiply(double[][] matrix, double[] vector) {
        double[] result = new double[vector.length];
        for (int row = 0; row < matrix.length; row++) {
            for (int column = 0; column < vector.length; column++) {
                result[row] += matrix[row][column] * vector[column];
            }
        }
        return result;
    }

    private static double[] orthogonalize(double[] candidate, double[] axis) {
        if (candidate.length == 0) {
            return candidate;
        }
        double projection = VectorMath.dot(candidate, axis);
        return VectorMath.subtract(candidate, VectorMath.scale(axis, projection));
    }

    private static double[] fallbackAxis(int dimensions, double[] axisX) {
        for (int i = 0; i < dimensions; i++) {
            double[] candidate = canonicalAxis(dimensions, i);
            double[] orthogonal = orthogonalize(candidate, axisX);
            if (norm(orthogonal) > EPSILON) {
                return orthogonal;
            }
        }
        return new double[dimensions];
    }

    private static double[] canonicalAxis(int dimensions, int index) {
        double[] axis = new double[dimensions];
        if (dimensions > 0) {
            axis[Math.min(index, dimensions - 1)] = 1.0;
        }
        return axis;
    }

    private static double norm(double[] vector) {
        return Math.sqrt(VectorMath.dot(vector, vector));
    }
}
