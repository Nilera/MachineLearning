package com.ifmo.machinelearning.library.core;

/**
 * Created by warrior on 05.12.14.
 */
public interface Instance {

    int getAttributeNumber();

    String getAttributeName(int i);

    double getAttributeValue(int i);

    double[] getValues();
}
