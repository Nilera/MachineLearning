package com.ifmo.machinelearning.library.feature.extraction;

import com.ifmo.machinelearning.library.core.Instance;

import java.util.List;

/**
 * Created by Ivan Samborskiy on 11/6/2016.
 */
public abstract class FeatureExtractor {

    protected final List<Instance> instances;

    public FeatureExtractor(List<Instance> instances) {
        this.instances = instances;
    }

    public abstract List<Instance> extract();

//    public abstract Instance extract(Instance instance);
}
