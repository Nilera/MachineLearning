package com.ifmo.machinelearning.library.ranker;

import com.ifmo.machinelearning.library.core.ClassifiedData;
import com.ifmo.machinelearning.library.core.ClassifiedInstance;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Ivan Samborskiy on 11/3/2016.
 */
public abstract class AbstractRanker {

    public List<Integer> rank(List<ClassifiedInstance> data) {
        double[] values = IntStream.range(0, data.get(0).getAttributeNumber())
                .mapToDouble(i -> rank(data, i))
                .toArray();

        return IntStream.range(0, data.get(0).getAttributeNumber())
                .boxed()
                .sorted((o1, o2) -> -Double.compare(values[o1], values[o2]))
                .collect(Collectors.toList());
    }

    protected abstract double rank(List<ClassifiedInstance> data, int index);

    protected boolean uniformArray(double[] array) {
        return Arrays.stream(array).noneMatch(value -> value != array[0]);
    }

    protected double[] values(List<ClassifiedInstance> data, int index) {
        return data.stream()
                .mapToDouble(instance -> instance.getAttributeValue(index))
                .toArray();
    }

    protected double[] classValues(List<ClassifiedInstance> data) {
        return data.stream()
                .mapToDouble(ClassifiedData::getClassId)
                .toArray();
    }
}
