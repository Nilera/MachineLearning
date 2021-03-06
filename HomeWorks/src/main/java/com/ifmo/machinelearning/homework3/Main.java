package com.ifmo.machinelearning.homework3;

import com.ifmo.machinelearning.library.core.ClassifiedInstance;
import com.ifmo.machinelearning.library.core.InstanceCreator;
import com.ifmo.machinelearning.library.test.Statistics;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by Whiplash on 05.10.2014.
 */
public class Main {

    private static final int FOLD_NUMBER = 5;
    private static final int ROUNDS = 50;

    public static void main(String[] args) throws IOException {
        List<ClassifiedInstance> sample = InstanceCreator.classifiedInstancesFromFile("./HomeWorks/res/homework3/LinearDataset");
        Collections.shuffle(sample);
        List<ClassifiedInstance> first = new ArrayList<>(sample.subList(0, sample.size() / 5));
        List<ClassifiedInstance> second = new ArrayList<>(sample.subList(sample.size() / 5, sample.size()));

        SVMTestMachine testMachine = new SVMTestMachine(second, true);
        double neededC = 0;
        double maxFMeasure = 0;
        for (int cPow = -5; cPow <= 11; cPow += 2) {
            double c = StrictMath.pow(2, cPow);
            testMachine.setC(c);
            Statistics statistics = testMachine.crossValidationTest(FOLD_NUMBER, ROUNDS);
            double fMeasure = statistics.getFMeasure();
            if (maxFMeasure < fMeasure) {
                maxFMeasure = fMeasure;
                neededC = c;
            }
            System.out.println("F-Measure " + fMeasure + " --- with C = " + c);
        }
        testMachine.setC(neededC);
        Statistics statistics = testMachine.test(first);
        System.out.println(statistics.getFMeasure());
    }
}
