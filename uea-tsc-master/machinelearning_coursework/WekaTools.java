/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning_coursework;

import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author yqm15fqu
 */
public class WekaTools {

    public static double measureAccuracy(Classifier c, Instances test) throws Exception {
        int counter = 0;

        for (Instance instance : test) {
            if (c.classifyInstance(instance) == instance.classValue()) {
                counter++;
            }
        }

//        System.out.println("Correctness: " + counter);
        double numInst = test.numInstances();

        System.out.println("Accuracy: " + ((double) counter / numInst) * 100);

        double accuracy = ((counter / numInst) * 100);

        return accuracy;
    }

    public static Instances loadClassificationData(String fullPath) {
        String dataLocation = fullPath;

        Instances train = null;
        try {
            FileReader reader = new FileReader(dataLocation);
            train = new Instances(reader);
        } catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }
        train.setClassIndex(train.numAttributes() - 1);
        return train;
    }

    public static Instances[] splitData(Instances all, double proportion) {
        Instances[] split = new Instances[2];
        Random random = new Random();
        split[0] = new Instances(all);
        split[1] = new Instances(all, 0);
        split[0].randomize(random);
        
        int splitAmount = (int) Math.floor((proportion * all.numInstances()));
        
        for (int i = 0; i < splitAmount; i++) {
            split[1].add(split[0].remove(0));
        }
//        System.out.println(split[0].size());
//        System.out.println(split[1].size());
        return split;
    }

    public static double[] classDistribution(Instances data) {
        double[] temp = new double[data.numClasses()];
        for (int i = 0; i < data.numInstances(); i++) {
            temp[(int) data.get(i).classValue()] += 1;
        }

        for (int i = 0; i < temp.length; i++) {
            temp[i] = temp[i] / data.numInstances();
        }
        return temp;
    }

    public static void main(String[] args) throws Exception {
        Instances train = loadClassificationData("//ueahome4/stusci1/yqm15fqu/data/Documents/MachineLearning_lab_2new/src/machinelearning_lab_2new/Arsenal_TRAIN.arff");

        double[] test = classDistribution(train);
        System.out.println(Arrays.toString(test));
    }

    public static int[][] confusionMatrix(int[] predicted, int[] actual) {
        int[][] confMatrix = new int[2][2];
        for (int i = 0; i < predicted.length; i++) {
            if (predicted[i] == 0) {
                if (actual[i] == predicted[i]) {
                    confMatrix[0][0] += 1;
                } else {
                    confMatrix[0][1] += 1;
                }
            } else if (actual[i] == predicted[i]) {
                confMatrix[1][1] += 1;
            } else {
                confMatrix[1][0] += 1;
            }
        }
        return confMatrix;
    }

    public static String[] listFilesForFolder(File folder) {
        File[] files = folder.listFiles();
        String[] bla = new String[files.length];
        int as = 0;
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFilesForFolder(fileEntry);
            } else {
                bla[as] = fileEntry.getName();
                as++;
                System.out.println(fileEntry.getName());
            }
        }
        return bla;
    }

    public static int[] classifyInstances(Classifier c, Instances test) throws Exception {
        int[] r = new int[test.size()];
        for (int i = 0; i < test.size(); i++) {
            r[i] = (int) c.classifyInstance(test.get(i));
        }
        return r;
    }

    public static int[] getClassValues(Instances data) {
        int[] classValues = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            classValues[i] = (int) data.get(i).classValue();
        }

        return classValues;
    }

}
