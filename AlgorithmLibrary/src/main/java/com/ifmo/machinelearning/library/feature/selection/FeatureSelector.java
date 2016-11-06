package com.ifmo.machinelearning.library.feature.selection;

import com.ifmo.machinelearning.library.core.ClassifiedInstance;
import com.ifmo.machinelearning.library.core.ClassifiedSubinstance;
import com.ifmo.machinelearning.library.core.TrainingAlgorithm;
import com.ifmo.machinelearning.library.ranker.AbstractRanker;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by Ivan Samborskiy on 11/4/2016.
 */
public class FeatureSelector implements TrainingAlgorithm {

    private final AbstractRanker ranker;

    private List<ClassifiedInstance> data;
    private List<Integer> orderedAttrs;

    public FeatureSelector(AbstractRanker ranker) {
        this.ranker = ranker;
    }

    public FeatureSelector setData(List<ClassifiedInstance> data) {
        this.data = data;
        return this;
    }

    public List<Integer> getOrderedAttrs() {
        return orderedAttrs;
    }

    @Override
    public FeatureSelector train() {
        this.orderedAttrs = ranker.rank(data);
        return this;
    }

    public List<ClassifiedInstance> select(int featuresNumber) {
        int[] filter = orderedAttrs.stream().limit(featuresNumber).mapToInt(i -> i).toArray();
        return data.stream().map(instance -> new ClassifiedSubinstance(instance, filter)).collect(Collectors.toList());
    }

    public List<ClassifiedInstance> prepareTest(int featuresNumber, List<ClassifiedInstance> test) {
        int[] filter = orderedAttrs.stream().limit(featuresNumber).mapToInt(i -> i).toArray();
        return test.stream().map(instance -> new ClassifiedSubinstance(instance, filter)).collect(Collectors.toList());
    }
}
