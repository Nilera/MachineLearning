package com.ifmo.machinelearning.library.classifiers.trees;

import com.ifmo.machinelearning.library.classifiers.AbstractInstanceClassifier;
import com.ifmo.machinelearning.library.core.ClassifiedInstance;
import com.ifmo.machinelearning.library.core.Instance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Created by Whiplash on 02.12.2014.
 */
public class DecisionTree extends AbstractInstanceClassifier {

    private Tree root;
    private QualityCriterion criterion;
    private int stopSize;
    private int bestErrorCount;

    private boolean pruning = true;

    public DecisionTree(List<ClassifiedInstance> data, QualityCriterion criterion, int stopSize) {
        super(data);
        this.criterion = criterion;
        this.stopSize = stopSize;
    }

    public void enablePruning(boolean enable) {
        this.pruning = enable;
    }

    @Override
    protected void trainInternal() {
        Collections.shuffle(getData());
        if (pruning) {
            int board = getData().size() * 2 / 3;
            List<ClassifiedInstance> train = getData().subList(0, board);
            List<ClassifiedInstance> control = getData().subList(board, getData().size());
            root = buildTree(null, train);
            bestErrorCount = checkTree(control);

            int oldErrorCount = bestErrorCount;
            int pruningCount = 0;
            while (pruning(root, control)) {
                pruningCount++;
            }
            int newErrorCount = checkTree(control);
            System.out.println("----------------------");
            System.out.println("Number of pruning: " + pruningCount);
            System.out.println("Error count: " + oldErrorCount + " -> " + newErrorCount);
        } else {
            root = buildTree(null, getData());
        }
    }

    private boolean pruning(Tree tree, List<ClassifiedInstance> control) {
        boolean isImprove = false;
        if (tree.getClassId() != Tree.NO_ENTRY_VALUE) {
            return false;
        } else {
            for (int i = 0; i < tree.size(); i++) {
                isImprove |= pruning(tree.getChild(i), control);
            }
            if (!isImprove && tree.getParent() != null) {
                int index = -1;
                for (int i = 0; i < tree.getParent().size(); i++) {
                    if (tree.getParent().getChild(i).equals(tree)) {
                        index = i;
                        break;
                    }
                }
                for (int i = 0; i < tree.size(); i++) {
                    isImprove = modifyTree(index, tree, tree.getChild(i), control);
                    if (isImprove) {
                        break;
                    }
                }
            }
        }
        return isImprove;
    }

    private boolean modifyTree(int index, Tree tree, Tree child, List<ClassifiedInstance> control) {
        tree.getParent().setChild(index, child);
        int errorCount = checkTree(control);
        if (bestErrorCount > errorCount) {
            bestErrorCount = errorCount;
            child.setParent(tree.getParent());
            return true;
        } else {
            tree.getParent().setChild(index, tree);
            return false;
        }
    }

    private int checkTree(List<ClassifiedInstance> instances) {
        int errorCount = 0;
        for (ClassifiedInstance instance : instances) {
            Tree node = root;
            while (node.getClassId() == Tree.NO_ENTRY_VALUE) {
                node = node.getChild(node.eval(instance));
            }
            if (node.getClassId() != instance.getClassId()) {
                errorCount++;
            }
        }
        return errorCount;
    }

    private Tree buildTree(Tree parent, List<ClassifiedInstance> sample) {
        if (sample.size() > stopSize) {
            boolean isSameClass = true;
            boolean isSameValues = true;

            ClassifiedInstance instance = sample.get(0);
            int classId = instance.getClassId();

            for (int i = 1; i < sample.size() && (isSameClass || isSameValues); i++) {
                ClassifiedInstance currentInstance = sample.get(i);
                isSameClass &= (classId == currentInstance.getClassId());
                isSameValues &= haveSameValues(instance, currentInstance);
            }

            if (!isSameClass && !isSameValues) {
                Tree tree = splitSample(sample);
                tree.setParent(parent);
                return tree;
            }

        }
        return new Tree(getMostPopularClassId(sample));
    }

    private boolean haveSameValues(Instance first, Instance second) {
        for (int i = 0; i < first.getAttributeNumber(); i++) {
            if (first.getAttributeValue(i) != second.getAttributeValue(i)) {
                return false;
            }
        }
        return true;
    }

    private Tree splitSample(List<ClassifiedInstance> sample) {
        int attributeNumber = sample.get(0).getAttributeNumber();
        double bestQuality = -1;
        Function<ClassifiedInstance, Integer> bestFunction = null;
        for (int i = 0; i < attributeNumber; i++) {
            int attribute = i;
            Set<Double> set = sample.stream()
                    .map(instance -> instance.getAttributeValue(attribute))
                    .collect(Collectors.toSet());
            List<Double> values = new ArrayList<>(set);
            Collections.sort(values);
            for (int j = 0; j < values.size() - 1; j++) {
                double value = (values.get(j) + values.get(j + 1)) / 2;
                Function<ClassifiedInstance, Integer> function = instance -> instance.getAttributeValue(attribute) > value ? 1 : 0;
                double quality = criterion.getValue(splitInstances(function, sample));
                if (bestQuality < quality) {
                    bestFunction = function;
                    bestQuality = quality;
                }
            }
        }

        List<List<ClassifiedInstance>> lists = splitInstances(bestFunction, sample);
        Tree tree = new Tree(bestFunction, null);
        List<Tree> children = lists.stream()
                .map(list -> buildTree(tree, list))
                .collect(Collectors.toList());
        tree.setChildren(children);
        return tree;
    }

    private int getMostPopularClassId(List<ClassifiedInstance> instances) {
        Map<Integer, Integer> classIds = new HashMap<>(instances.get(0).getClassNumber());
        for (ClassifiedInstance instance : instances) {
            int id = instance.getClassId();
            classIds.put(id, classIds.getOrDefault(id, 0) + 1);
        }
        int bestId = 0;
        int bestEntries = -1;
        for (Integer classId : classIds.keySet()) {
            int entries = classIds.get(classId);
            if (entries > bestEntries) {
                bestEntries = entries;
                bestId = classId;
            }
        }
        return bestId;
    }

    private List<List<ClassifiedInstance>> splitInstances(Function<ClassifiedInstance, Integer> function,
                                                          List<ClassifiedInstance> sample) {
        Map<Integer, List<ClassifiedInstance>> map = new HashMap<>();
        List<List<ClassifiedInstance>> lists = new ArrayList<>(getClassNumber());
        for (ClassifiedInstance instance : sample) {
            Integer value = function.apply(instance);
            List<ClassifiedInstance> list = map.getOrDefault(value, new ArrayList<>());
            list.add(instance);
            map.put(value, list);
        }
        lists.addAll(map.keySet().stream().map(map::get).collect(Collectors.toList()));
        return lists;
    }

    @Override
    public int getSupposedClassId(ClassifiedInstance classifiedInstance) {
        Tree node = root;
        while (node.getClassId() == Tree.NO_ENTRY_VALUE) {
            node = node.getChild(node.eval(classifiedInstance));
        }
        return node.getClassId();
    }
}
