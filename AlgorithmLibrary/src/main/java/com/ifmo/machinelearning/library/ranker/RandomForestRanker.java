package com.ifmo.machinelearning.library.ranker;

import com.ifmo.machinelearning.library.classifiers.trees.DecisionTree;
import com.ifmo.machinelearning.library.classifiers.trees.QualityCriterion;
import com.ifmo.machinelearning.library.core.ClassifiedInstance;
import com.ifmo.machinelearning.library.core.ClassifiedInstanceDefaultImpl;
import com.ifmo.machinelearning.library.core.ClassifiedSubinstance;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RandomForestRanker extends AbstractRanker {

    private static final Random RANDOM = new Random();

    private final int treeDepth;
    private final int treesNumber;
    private final QualityCriterion criterion;

    private int featuresNumber;
    private int dtFeaturesNumber;

    public RandomForestRanker(int treeDepth, int treesNumber, QualityCriterion criterion) {
        this.treeDepth = treeDepth;
        this.treesNumber = treesNumber;
        this.criterion = criterion;
    }

    @Override
    public List<Integer> rank(List<ClassifiedInstance> data) {
        featuresNumber = data.get(0).getAttributeNumber();
        dtFeaturesNumber = (int) Math.sqrt(data.get(0).getAttributeNumber());

        List<DecisionTreeWrapper> randomForest = buildRandomForest(data);
        double[] errors = calculateErrors(randomForest, data);

        String[] attributes = IntStream.range(0, featuresNumber)
                .mapToObj(i -> data.get(0).getAttributeName(i))
                .toArray(String[]::new);

        double[] attrError = IntStream.range(0, featuresNumber).mapToDouble(index -> {
            List<Double> values = data.stream()
                    .map(instance -> instance.getAttributeValue(index))
                    .collect(Collectors.toList());
            Collections.shuffle(values, RANDOM);
            List<ClassifiedInstance> modifiedData = IntStream.range(0, data.size())
                    .mapToObj(i -> {
                        ClassifiedInstance instance = data.get(i);
                        double[] vs = instance.getValues();
                        vs[index] = values.get(i);
                        return new ClassifiedInstanceDefaultImpl(attributes, vs, instance.getClassNumber(), instance.getClassId());
                    })
                    .collect(Collectors.toList());

            double[] newErrors = calculateErrors(randomForest, modifiedData);
            double[] errorDif = IntStream.range(0, newErrors.length)
                    .mapToDouble(i -> errors[i] - newErrors[i])
                    .toArray();

            double error = 0;
            if (!uniformArray(newErrors)) {
                double mean = Arrays.stream(errorDif).average().getAsDouble();
                double stdDeviations = Math.sqrt(variance(errorDif, mean));
                error = mean / stdDeviations;
            }
            return error;
        }).toArray();

        return IntStream.range(0, featuresNumber)
                .boxed()
                .sorted((o1, o2) -> -Double.compare(attrError[o1], attrError[o2]))
                .collect(Collectors.toList());
    }

    private List<DecisionTreeWrapper> buildRandomForest(List<ClassifiedInstance> data) {
        return IntStream.range(0, treesNumber)
                .mapToObj(i -> buildDecisionTree(data))
                .collect(Collectors.toList());
    }

    private DecisionTreeWrapper buildDecisionTree(List<ClassifiedInstance> data) {
        List<Integer> indices = IntStream.range(0, featuresNumber).boxed().collect(Collectors.toList());
        Collections.shuffle(indices, RANDOM);
        int[] filter = indices.stream().limit(dtFeaturesNumber).mapToInt(i -> i).toArray();
        Set<Integer> indicesOfInstances = new HashSet<>();

        List<ClassifiedInstance> subset = IntStream.range(0, data.size())
                .mapToObj(i -> {
                    int index = RANDOM.nextInt(data.size());
                    indicesOfInstances.add(index);
                    return new ClassifiedSubinstance(data.get(index), filter);
                })
                .collect(Collectors.toList());

        DecisionTree tree = new DecisionTree(subset, criterion, treeDepth);
        tree.enablePruning(false);
        tree.train();
        return new DecisionTreeWrapper(tree, filter, indicesOfInstances);
    }

    private void applyAttributeChanges(List<ClassifiedInstance> data, double[] values, int attributeIndex) {
        IntStream.range(0, data.size())
                .forEach(i -> data.get(i).getValues()[attributeIndex] = values[i]);
    }

    private double[] calculateErrors(List<DecisionTreeWrapper> randomForest, List<ClassifiedInstance> data) {
        return IntStream.range(0, data.size())
                .mapToDouble(i -> calculateOutOfBag(randomForest, data.get(i), i))
                .toArray();
    }

    private double calculateOutOfBag(List<DecisionTreeWrapper> randomForest, ClassifiedInstance instance, int index) {
        double error = 0;
        double all = 0;
        for (DecisionTreeWrapper wrapper : randomForest) {
            if (!wrapper.indicesOfInstances.contains(index)) {
                all++;
                ClassifiedInstance subinstance = new ClassifiedSubinstance(instance, wrapper.filter);
                if (subinstance.getClassId() != wrapper.tree.getSupposedClassId(subinstance)) {
                    error++;
                }
            }
        }
        return error / all;
    }

    private double variance(double[] values, double mean) {
        if (values.length == 1) {
            return 0;
        }
        return Arrays.stream(values)
                .map(v -> (v - mean) * (v - mean))
                .sum() / (values.length - 1);
    }

    @Override
    protected double rank(List<ClassifiedInstance> data, int index) {
        throw new NotImplementedException();
    }

    private static class DecisionTreeWrapper {

        public final DecisionTree tree;
        public final int[] filter;
        public final Set<Integer> indicesOfInstances;

        public DecisionTreeWrapper(DecisionTree tree, int[] filter, Set<Integer> indicesOfInstances) {
            this.tree = tree;
            this.filter = filter;
            this.indicesOfInstances = indicesOfInstances;
        }
    }
}
