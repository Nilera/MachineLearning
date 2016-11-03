package com.ifmo.machinelearning.library.util;

import com.ifmo.machinelearning.library.core.ClassifiedInstance;
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
}
