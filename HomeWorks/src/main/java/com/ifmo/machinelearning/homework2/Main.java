package com.ifmo.machinelearning.homework2;

import com.ifmo.machinelearning.graphics.Plot2DBuilder;
import com.ifmo.machinelearning.test.Statistics;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by Whiplash on 28.09.2014.
 */
public class Main {

    private static final File messageDirectory = new File("./Bayes/pu1");

    private static final int FOLD_NUMBER = 5;
    private static final int ROUNDS = 100;

    private static List<Message> sample;

    public static void main(String[] args) throws IOException {
        sample = new ArrayList<>();
        getSamples(messageDirectory);
        Collections.shuffle(sample);
        BayesTestMachine testMachine = new BayesTestMachine(sample, false);
        Statistics statistics = testMachine.crossValidationTest(FOLD_NUMBER, ROUNDS);
        System.out.println(statistics.getFMeasure());
    }

    private static void getSamples(File file) throws IOException {
        if (file.isDirectory()) {
            for (File f : file.listFiles()) {
                getSamples(f);
            }
        } else {
            sample.add(Message.createMessage(file));
        }
    }
}