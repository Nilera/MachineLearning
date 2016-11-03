package com.ifmo.machinelearning.library.ranker;

import com.ifmo.machinelearning.library.core.ClassifiedInstance;
import com.ifmo.machinelearning.library.util.InstancesUtil;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.core.Instances;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by Ivan Samborskiy on 11/3/2016.
 */
public class MutialInformationRanker extends AbstractRanker {

    @Override
    public List<Integer> rank(List<ClassifiedInstance> data) {
        try {
            Instances instances = InstancesUtil.toInstances(data);
            InfoGainAttributeEval infoGainAttributeEval = new InfoGainAttributeEval();
            infoGainAttributeEval.buildEvaluator(instances);

            Map<Integer, Double> iGains = new HashMap<>();
            for (int i = 0; i < instances.numAttributes() - 1; i++) {
                double iGain = infoGainAttributeEval.evaluateAttribute(i);
                iGains.put(i, iGain);
            }
            return iGains.keySet().stream()
                    .filter(index -> index != instances.classIndex())
                    .sorted((o1, o2) -> -Double.compare(iGains.get(o1), iGains.get(o2)))
                    .collect(Collectors.toList());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected double rank(List<ClassifiedInstance> data, int index) {
        throw new NotImplementedException();
    }
}
