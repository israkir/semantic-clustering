package dev.semanticclustering.util;

import java.util.Arrays;
import java.util.List;

public final class VectorMath {
    private VectorMath() {
    }

    public static double[] l2Normalize(double[] vector) {
        double[] copy = Arrays.copyOf(vector, vector.length);
        double norm = 0.0;
        for (double value : copy) {
            norm += value * value;
        }
        norm = Math.sqrt(norm);
        if (norm == 0.0) {
            return copy;
        }
        for (int i = 0; i < copy.length; i++) {
            copy[i] /= norm;
        }
        return copy;
    }

    public static float[] l2Normalize(float[] vector) {
        float[] copy = Arrays.copyOf(vector, vector.length);
        double norm = 0.0;
        for (float value : copy) {
            norm += value * value;
        }
        norm = Math.sqrt(norm);
        if (norm == 0.0) {
            return copy;
        }
        for (int i = 0; i < copy.length; i++) {
            copy[i] = (float) (copy[i] / norm);
        }
        return copy;
    }

    public static double dot(double[] left, double[] right) {
        double sum = 0.0;
        for (int i = 0; i < left.length; i++) {
            sum += left[i] * right[i];
        }
        return sum;
    }

    public static double[] subtract(double[] left, double[] right) {
        double[] result = new double[left.length];
        for (int i = 0; i < left.length; i++) {
            result[i] = left[i] - right[i];
        }
        return result;
    }

    public static double[] add(double[] left, double[] right) {
        double[] result = new double[left.length];
        for (int i = 0; i < left.length; i++) {
            result[i] = left[i] + right[i];
        }
        return result;
    }

    public static double[] scale(double[] vector, double factor) {
        double[] result = Arrays.copyOf(vector, vector.length);
        for (int i = 0; i < result.length; i++) {
            result[i] *= factor;
        }
        return result;
    }

    public static double[] mean(double[][] vectors) {
        if (vectors.length == 0) {
            return new double[0];
        }
        double[] mean = new double[vectors[0].length];
        for (double[] vector : vectors) {
            for (int i = 0; i < mean.length; i++) {
                mean[i] += vector[i];
            }
        }
        for (int i = 0; i < mean.length; i++) {
            mean[i] /= vectors.length;
        }
        return mean;
    }

    public static double[] mean(double[][] vectors, List<Integer> indexes) {
        if (indexes.isEmpty()) {
            return new double[0];
        }
        double[] mean = new double[vectors[indexes.get(0)].length];
        for (int index : indexes) {
            double[] vector = vectors[index];
            for (int i = 0; i < mean.length; i++) {
                mean[i] += vector[i];
            }
        }
        for (int i = 0; i < mean.length; i++) {
            mean[i] /= indexes.size();
        }
        return mean;
    }
}
