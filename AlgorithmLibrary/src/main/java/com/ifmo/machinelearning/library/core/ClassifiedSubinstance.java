package com.ifmo.machinelearning.library.core;

import java.util.Arrays;

/**
 * Created by Ivan Samborskiy on 11/4/2016.
 */
public class ClassifiedSubinstance implements ClassifiedInstance {

    private final ClassifiedInstance instance;
    private final int[] filter;

    public ClassifiedSubinstance(ClassifiedInstance instance, int[] filter) {
        this.instance = instance;
        this.filter = filter;
    }

    @Override
    public int getAttributeNumber() {
        return filter.length;
    }

    @Override
    public String getAttributeName(int i) {
        return instance.getAttributeName(filter[i]);
    }

    @Override
    public double getAttributeValue(int i) {
        return instance.getAttributeValue(filter[i]);
    }

    @Override
    public double[] getValues() {
        return Arrays.stream(filter).mapToDouble(instance::getAttributeValue).toArray();
    }

    @Override
    public int getClassId() {
        return instance.getClassId();
    }

    @Override
    public int getClassNumber() {
        return instance.getClassNumber();
    }
}
