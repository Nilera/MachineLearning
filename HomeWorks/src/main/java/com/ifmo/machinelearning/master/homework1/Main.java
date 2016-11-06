package com.ifmo.machinelearning.master.homework1;

import com.ifmo.machinelearning.library.classifiers.trees.GiniGain;
import com.ifmo.machinelearning.library.core.ClassifiedInstance;
import com.ifmo.machinelearning.library.core.InstanceCreator;
import com.ifmo.machinelearning.library.feature.selection.FeatureSelector;
import com.ifmo.machinelearning.library.ranker.*;
import com.ifmo.machinelearning.library.util.InstancesUtil;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class Main {

    private static final Classifier CLASSIFIER = new IBk(3);

    private static final int FEATURES_NUMBER_UPPER_BOUND = 100;

    private static final AbstractRanker[] RANKERS = {
            new RandomForestRanker(5, 200, new GiniGain()),
            new MutialInformationRanker(),
            new PearsonsCorrelationRanker(),
            new SpearmansCorrelationRanker()
    };

    private static final String TRAIN_FILE = "res/master-homework1/train";
    private static final String TEST_FILE = "res/master-homework1/valid";

    public static void main(String[] args) throws Exception {
        List<ClassifiedInstance> train = InstanceCreator.classifiedInstancesFromFile(TRAIN_FILE);
        List<ClassifiedInstance> test = InstanceCreator.classifiedInstancesFromFile(TEST_FILE);

        for (AbstractRanker ranker : RANKERS) {
            List<Double> results = new ArrayList<>();
            FeatureSelector selector = new FeatureSelector(ranker).setData(train).train();
            for (int index = 1; index < FEATURES_NUMBER_UPPER_BOUND; index++) {
                List<ClassifiedInstance> trainClassifiedInstances = selector.select(index);
                List<ClassifiedInstance> testClassifiedInstances = selector.prepareTest(index, test);

                Instances trainInstances = InstancesUtil.toInstances(trainClassifiedInstances);
                Instances testInstances = InstancesUtil.toInstances(testClassifiedInstances);
                CLASSIFIER.buildClassifier(trainInstances);

                Evaluation evaluation = new Evaluation(testInstances);
                evaluation.evaluateModel(CLASSIFIER, testInstances);
                double fMeasure = evaluation.unweightedMacroFmeasure();
                results.add(fMeasure);
            }

            try (PrintWriter writer = new PrintWriter(ranker.getClass().getSimpleName() + "_kNN3")) {
                results.forEach(writer::println);
            }
            try (PrintWriter writer = new PrintWriter(ranker.getClass().getSimpleName() + "_features")) {
                selector.getOrderedAttrs().stream().limit(FEATURES_NUMBER_UPPER_BOUND).forEach(writer::println);
            }
        }
    }
}
