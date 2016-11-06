package com.ifmo.machinelearning.library.feature.extraction;

import com.ifmo.machinelearning.library.core.Instance;
import com.ifmo.machinelearning.library.util.InstancesUtil;
import org.jblas.DoubleMatrix;
import org.jblas.Eigen;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by Ivan Samborskiy on 11/6/2016.
 */
public class PCA extends FeatureExtractor {

    private final String[] attributes;

    private DoubleMatrix transformationMatrix;
    private String[] modifiedAttributes;

    public PCA(List<Instance> instances) {
        super(instances);

        this.attributes = IntStream.range(0, instances.get(0).getAttributeNumber())
                .mapToObj(i -> instances.get(0).getAttributeName(i))
                .toArray(String[]::new);
    }

    @Override
    public List<Instance> extract() {
        int n = instances.get(0).getAttributeNumber();

        DoubleMatrix matrix = buildCorrelationMatrix(instances);
        DoubleMatrix[] eigenmatrix = Eigen.symmetricEigenvectors(matrix);

        double[] ascEigenvalues = IntStream.range(0, n)
                .mapToDouble(i -> eigenmatrix[1].get(i, i))
                .toArray();
        int featuresNumber = calculateBrokenStick(matrix, ascEigenvalues);

        transformationMatrix = new DoubleMatrix(n, featuresNumber);
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < featuresNumber; c++) {
                transformationMatrix.put(r, c, eigenmatrix[0].get(r, n - c - 1));
            }
        }
        modifiedAttributes = IntStream.range(0, featuresNumber)
                .mapToObj(i -> "attr" + i)
                .toArray(String[]::new);

        DoubleMatrix dataset = InstancesUtil.toMatrix(instances);
        DoubleMatrix modifiedDataset = dataset.mmul(transformationMatrix);
        return InstancesUtil.toInstances(modifiedAttributes, modifiedDataset);
    }

    private int calculateBrokenStick(DoubleMatrix matrix, double[] ascEigenvalues) {
        int n = ascEigenvalues.length;

        double trace = trace(matrix);
        double[] descNormEigenvalues = IntStream.range(0, n)
                .map(i -> n - i - 1)
                .mapToDouble(i -> ascEigenvalues[i] / trace)
                .toArray();
        double[] l = IntStream.rangeClosed(1, n)
                .mapToDouble(i -> IntStream.range(i, n)
                        .mapToDouble(j -> 1 / ((double) j))
                        .sum() / n)
                .toArray();

        for (int i = 0; i < n; i++) {
            if (descNormEigenvalues[i] < l[i]) {
                return i;
            }
        }

        throw new RuntimeException("broken stick method doesn't work");
    }

    private double trace(DoubleMatrix matrix) {
        double trace = 0.;
        for (int i = 0; i < matrix.getRows(); i++) {
            trace += matrix.get(i, i);
        }
        return trace;
    }

    private DoubleMatrix buildCorrelationMatrix(List<Instance> instances) {
        int n = instances.get(0).getAttributeNumber();
        DoubleMatrix matrix = new DoubleMatrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {

                double cov = 0;
                for (Instance instance : instances) {
                    cov += instance.getAttributeValue(i) * instance.getAttributeValue(j);
                }
                cov /= (instances.size() - 1);

                matrix.put(i, j, cov);
                matrix.put(j, i, cov);
            }
        }
        return matrix;
    }
}
