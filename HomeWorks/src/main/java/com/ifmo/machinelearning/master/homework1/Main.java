package com.ifmo.machinelearning.master.homework1;

import com.ifmo.machinelearning.library.core.ClassifiedInstance;
import com.ifmo.machinelearning.library.core.InstanceCreator;
import com.ifmo.machinelearning.library.ranker.*;
import com.ifmo.machinelearning.library.util.InstancesUtil;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.util.List;

public class Main {

    private static final Classifier[] CLASSIFIERS = {
            new IBk(3)
    };

    private static final AbstractRanker[] RANKERS = {
            new MutialInformationRanker(),
            new PearsonsCorrelationRanker(),
            new SpearmansCorrelationRanker()
    };

    private static final String TRAIN_FILE = "res/master-homework1/train";

    public static void main(String[] args) {
        List<ClassifiedInstance> train = InstanceCreator.classifiedInstancesFromFile(TRAIN_FILE);

//        for (AbstractRanker ranker : RANKERS) {
//            List<Integer> attributes = ranker.rank(train);
//            System.out.println(attributes.stream().map(String::valueOf).collect(Collectors.joining(" ")));
//        }
        FeatureSelection selector = new FeatureSelection(RANKERS[1]).setData(train).train();
        List<ClassifiedInstance> instances = selector.select(10);
        Instances instances1 = InstancesUtil.toInstances(instances);
    }
}
