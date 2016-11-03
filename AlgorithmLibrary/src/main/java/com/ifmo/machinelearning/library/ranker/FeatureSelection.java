package com.ifmo.machinelearning.library.ranker;

import com.ifmo.machinelearning.library.core.ClassifiedInstance;
import com.ifmo.machinelearning.library.core.ClassifiedSubinstance;
import com.ifmo.machinelearning.library.core.TrainingAlgorithm;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by Ivan Samborskiy on 11/4/2016.
 */
public class FeatureSelection implements TrainingAlgorithm {

    private final AbstractRanker ranker;

    private List<ClassifiedInstance> data;
    private List<Integer> orderedAttrs;

    public FeatureSelection(AbstractRanker ranker) {
        this.ranker = ranker;
    }

    public FeatureSelection setData(List<ClassifiedInstance> data) {
        this.data = data;
        return this;
    }

    @Override
    public FeatureSelection train() {
        this.orderedAttrs = ranker.rank(data);
        return this;
    }

    public List<ClassifiedInstance> select(int featuresNumber) {
        int[] filter = orderedAttrs.stream().limit(featuresNumber).mapToInt(i -> i).toArray();
        return data.stream().map(instance -> new ClassifiedSubinstance(instance, filter)).collect(Collectors.toList());
    }
}
