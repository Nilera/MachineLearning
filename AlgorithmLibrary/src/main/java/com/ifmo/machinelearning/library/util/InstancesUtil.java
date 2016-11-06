package com.ifmo.machinelearning.library.util;

import com.ifmo.machinelearning.library.core.ClassifiedInstance;
import com.ifmo.machinelearning.library.core.InstanceDefaultImpl;
import org.jblas.DoubleMatrix;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Ivan Samborskiy on 11/3/2016.
 */
public class InstancesUtil {

    private InstancesUtil() {
    }

    public static Instances toInstances(List<ClassifiedInstance> data) {
        ClassifiedInstance sample = data.get(0);
        ArrayList<Attribute> attributes = IntStream.range(0, sample.getAttributeNumber())
                .mapToObj(sample::getAttributeName)
                .map(Attribute::new)
                .collect(Collectors.toCollection(ArrayList::new));
        attributes.add(new Attribute("class", IntStream.range(0, sample.getClassNumber())
                .mapToObj(String::valueOf)
                .collect(Collectors.toList())));

        Instances instances = new Instances("", attributes, data.size());
        instances.setClassIndex(attributes.size() - 1);
        for (ClassifiedInstance inst : data) {
            Instance instance = new DenseInstance(attributes.size());
            for (int i = 0; i < attributes.size() - 1; i++) {
                instance.setValue(attributes.get(i), inst.getAttributeValue(i));
            }
            instance.setValue(attributes.size() - 1, inst.getClassId());
            instances.add(instance);
        }
        return instances;
    }

    public static DoubleMatrix toMatrix(List<com.ifmo.machinelearning.library.core.Instance> instances) {
        DoubleMatrix matrix = new DoubleMatrix(instances.size(), instances.get(0).getAttributeNumber());
        for (int r = 0; r < matrix.getRows(); r++) {
            com.ifmo.machinelearning.library.core.Instance instance = instances.get(r);
            for (int c = 0; c < matrix.getColumns(); c++) {
                matrix.put(r, c, instance.getAttributeValue(c));
            }
        }
        return matrix;
    }

    public static List<com.ifmo.machinelearning.library.core.Instance> toInstances(String[] attributes, DoubleMatrix matrix) {
        List<com.ifmo.machinelearning.library.core.Instance> instances = new ArrayList<>();
        for (int i = 0; i < matrix.getRows(); i++) {
            int r = i;
            double[] values = IntStream.range(0, matrix.getColumns())
                    .mapToDouble(c -> matrix.get(r, c))
                    .toArray();
            instances.add(toInstance(attributes, values));
        }
        return instances;
    }

    public static com.ifmo.machinelearning.library.core.Instance toInstance(String[] attributes, DoubleMatrix row) {
        double[] values = IntStream.range(0, row.getColumns())
                .mapToDouble(c -> row.get(0, c))
                .toArray();
        return toInstance(attributes, values);
    }

    public static com.ifmo.machinelearning.library.core.Instance toInstance(String[] attributes, double[] values) {
        return new InstanceDefaultImpl(attributes, values);
    }
}
