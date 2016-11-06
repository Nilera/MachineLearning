package com.ifmo.machinelearning.master.homework2;

import com.ifmo.machinelearning.library.core.Instance;
import com.ifmo.machinelearning.library.core.InstanceCreator;
import com.ifmo.machinelearning.library.feature.extraction.PCA;

import java.util.List;

/**
 * Created by Ivan Samborskiy on 11/6/2016.
 */
public class Main {

    private static final String[] DATASET_FILENAMES = {
            "res/master-homework2/newBasis1",
            "res/master-homework2/newBasis2",
            "res/master-homework2/newBasis3"
    };

    public static void main(String[] args) {
        for (String filename : DATASET_FILENAMES) {
            List<Instance> instances = InstanceCreator.instancesFromFile(filename);
            List<Instance> modifiedInstances = new PCA(instances).extract();
            System.out.println(modifiedInstances.get(0).getAttributeNumber());
        }
    }
}
