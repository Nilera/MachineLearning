package com.ifmo.machinelearning.library.ranker;

import com.ifmo.machinelearning.library.core.ClassifiedInstance;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

import java.util.List;

/**
 * Created by Ivan Samborskiy on 11/3/2016.
 */
public class PearsonsCorrelationRanker extends AbstractRanker {

    private static final PearsonsCorrelation PEARSONS_CORRELATION = new PearsonsCorrelation();

    @Override
    protected double rank(List<ClassifiedInstance> data, int index) {
        double[] values = values(data, index);
        double[] classValues = classValues(data);

        if (uniformArray(values) || uniformArray(classValues)) {
            return 0.;
        }
        return PEARSONS_CORRELATION.correlation(values, classValues);
    }
}
