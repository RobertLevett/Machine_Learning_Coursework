/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning_coursework;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;
import static weka.core.Utils.mean;

/**
 *
 * @author yqm15fqu
 */
public class KNN extends AbstractClassifier {

    private ArrayList<Instance> nearestNeighbours;
    private boolean standardiseAttributes = true;
    private boolean weightedVoting = false;
    private boolean leaveOneOut = false;
    private boolean draw = false;
    private Instances trainData;
    private double[] distances;
    private int maxK = 100;
    private int k = 1;

    public void setK(int x) {
        this.k = x;
    }

    public void setStandardise(boolean x) {
        this.standardiseAttributes = x;
    }

    public void setLeaveOneOut(boolean x) {
        this.leaveOneOut = x;
    }

    public void setWeightedVoting(boolean x) {
        this.weightedVoting = x;
    }

    public int getK() {
        return this.k;
    }

    public static double accuracy(Classifier c, Instances test) throws Exception {

        double result = 0;
        double accuracy = 0;
        int counter = 0;
        for (Instance instance : test) {
            result = c.classifyInstance(instance);
            if (result == instance.classValue()) {
                counter++;
            }
        }
        accuracy = (((double) counter / test.size()) * 100);

        return accuracy;
    }

    public static int mode(double[] input) {
        int[] count = new int[input.length + 1];
        //cont the occurrences
        for (int i = 0; i < input.length; i++) {
            count[(int) input[i]]++;
        }
        //go backwards and find the count with the most occurrences
        int index = count.length - 1;
        for (int i = count.length - 2; i >= 0; i--) {
            if (count[i] >= count[index]) {
                index = i;
            }
//            System.out.println(i + " is a count of " + count[i]);
        }

        return index;
    }

    private double[] kAccuracy(double[][] distArray) {
        double[][] allClassValues = new double[trainData.size()][maxK + 1];

        for (int i = 0; i < distArray.length; i++) {//loop through all rows
            ArrayList<InstanceContainer> allVals = new ArrayList<>(trainData.size());

            for (int l = 0; l < distArray.length; l++) {    // loop through each column in the current row
                InstanceContainer temporary = new InstanceContainer(trainData.get(l), distArray[i][l]);
                allVals.add(temporary);
            }
            //sort the arrayList by distances, so now have ascending order of distances and their respective instances
            Collections.sort(allVals);
            
            //get all instane class numbers, so now have a 2d array of ordered class numbers based on the instance distance.
            //each row of the 2d array represents the training data position that has the distances relate to.
            //for instance row allClassVals[3][0] is trainingData.get(3) and all values associated with it. 
            for (int j = 0; j <= this.maxK; j++) {
                allClassValues[i][j] = allVals.get(j).getInstance().classValue();
            }
        }
        double[][] kAccArray = new double[distArray.length][this.maxK + 1];

        for (int i = 0; i < trainData.size(); i++) {
            double classNum = trainData.get(i).classValue();
            //calculate accuracy
            for (int s = 1; s <= this.maxK; s++) {
                double totalPercent = 0;
                double[] rowClassVals = new double[trainData.numClasses()];

                for (int l = 1; l <= s; l++) {
                    rowClassVals[(int) allClassValues[i][l]]++;
                }


                totalPercent = rowClassVals[(int) classNum];
                totalPercent = (totalPercent / s) * 100;
                kAccArray[i][s] = totalPercent;
            }
        }

        double[] bestKValues = new double[trainData.size()];
        for (int i = 0; i < trainData.size(); i++) {
            //find the position that is highest and return it.
            double kVal = findHighestPos(kAccArray[i]);

            bestKValues[i] = kVal;
        }

        return bestKValues;
    }
    //find the mean
    public double mean(double[] accuracies) {
        double answer = 0;

        for (int i = 0; i < accuracies.length; i++) {
            answer += accuracies[i];
        }

        answer = answer / accuracies.length;

        return answer;
    }
    //find the highest pos
    private int findHighestPos(double[] x) {
        Random rand = new Random();
        double max = x[1];
        int value = 0;
        int doubleCounter = 0;
        double[] drawPos = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            if (x[i] > max) {
                draw = false;
                max = x[i];
                drawPos = new double[x.length];
                drawPos[0] = i;
                doubleCounter = 1;
            } else if (x[i] == max) {
                draw = true;
                drawPos[doubleCounter] = i;
                doubleCounter++;
            }
        }
        //randomly sorts draws
        if (draw) {
//            System.out.println("Draw! settled randomly."); //For debugging, remove later.
            double[] posArr = new double[doubleCounter];
            for (int i = 0; i < doubleCounter; i++) {
                posArr[i] = drawPos[i];
            }
            int chosen = rand.nextInt(posArr.length);
            value = (int) posArr[chosen];
            draw = false;
        } else {
            value = (int) drawPos[0];
        }

        return value;
    }
    //standardisation of attributes
    private void standardiseAttributes(Instances train) {
        for (int j = 0; j < this.trainData.numAttributes() - 1; j++) {
            double mean = 0;
            double stdDev = 0;
            for (Instance instance : this.trainData) {
                mean += instance.value(j);
            }
            mean = mean / this.trainData.size();
            for (int k = 0; k < this.trainData.size(); k++) {
                stdDev += Math.pow(this.trainData.get(k).value(j) - mean, 2);
            }
            stdDev = stdDev / this.trainData.size();
            stdDev = Math.sqrt(stdDev);
            for (Instance instance : this.trainData) {
                instance.setValue(j, ((instance.value(j) - mean) / stdDev));
//                  System.out.println(instance.value(j));
            }
        }
    }

    private int LOOCV() {
        //Set the size of maxK

        if (trainData.size() * 0.2 < 100) {
            this.maxK = (int) (this.trainData.size() * 0.2);
        } else {
            this.maxK = 100;
        }

        double[] kAccuracyArray = new double[this.maxK];
        // for all distances to all train data
        double[][] allDistances
                = new double[this.trainData.size()][this.trainData.size()];

        for (int j = 0; j < trainData.size(); j++) {
            Instance curInst = this.trainData.get(j);
            //done to populate the distances array
            double temp = classifyInstance(curInst);
            //stores each collection of distances into each row
            for (int l = 0; l < this.trainData.size(); l++) {
                if (l == j) {
                    allDistances[j][l] = -1;
                } else {
                    allDistances[j][l] = this.distances[l];
                }
            }
        }
        kAccuracyArray = kAccuracy(allDistances);
//        System.out.println(mode(kAccuracyArray));
//set k to the highest accuracy
        return mode(kAccuracyArray);
    }

    @Override
    public void buildClassifier(Instances i) {
        this.trainData = i;

        if (standardiseAttributes) {
            standardiseAttributes(trainData);
        }
        if (leaveOneOut) { //need to double check this as always returns 1. Moving on for now.
            this.k = LOOCV();

            System.out.println("LOOCV done, K set to: " + this.k);
        }
    }

    private double setNewWeight(double dist) {
        double newWeight = (double) 1 / (double) (1 + dist);
        return newWeight;
    }

    @Override
    public double classifyInstance(Instance test) {

        ArrayList<InstanceContainer> allVals = new ArrayList<>(this.trainData.size());
        double[] distRatio = new double[test.numClasses()];
        //Used to clear distances of the previously stored distances.
        this.distances = new double[trainData.size()];
        ArrayList<Instance> lowestFound = new ArrayList();

        //Populates the double array "lowest" with the first k instances of the training data
        //so that there are instances to compare the rest of the train data to.
        for (int i = 0; i < trainData.size(); i++) {
            distances[i] = euclidDist(test, trainData.get(i));
            InstanceContainer temp = new InstanceContainer(trainData.get(i), distances[i]);
            allVals.add(temp);
        }

        Collections.sort(allVals);

        for (int i = 0; i < k; i++) {
            lowestFound.add(allVals.get(i).getInstance());
        }

        nearestNeighbours = lowestFound;

        if (weightedVoting) {

            for (int i = 0; i < k; i++) {
                allVals.get(i).getInstance().setWeight(setNewWeight(allVals.get(i).getDistance()));
            }

            for (Instance instance : lowestFound) {
                distRatio[(int) instance.classValue()] += instance.weight();
            }
            return findHighestPos(distRatio);
        }

        for (int i = 0; i < lowestFound.size(); i++) {
            distRatio[(int) lowestFound.get(i).classValue()]++;
        }

//        if(findHighestPos(distRatio)!= test.classValue())
//            System.out.println("INCORRECT");
        return findHighestPos(distRatio);

    }

    public double euclidDist(Instance test, Instance train) {
        double[] distance = new double[train.numAttributes()];
        double sum = 0;

        for (int i = 0; i < train.numAttributes() - 1; i++) {
            distance[i] = Math.pow((test.value(i) - train.value(i)), 2);
        }
        for (int i = 0; i < distance.length; i++) {
            sum += (distance[i]);
        }
        return Math.sqrt(sum);
    }

    @Override
    public double[] distributionForInstance(Instance test) {

        double[] distRatio = new double[trainData.numClasses()];

        if (nearestNeighbours != null) {
            for (Instance instance : nearestNeighbours) {
                int classValue = (int) instance.classValue();
                distRatio[classValue]++;
            }

            for (int i = 0; i < distRatio.length; i++) {
                double x = distRatio[i];
                distRatio[i] = ((x / nearestNeighbours.size()));
            }
//        System.out.println(Arrays.toString(distRatio));
            return distRatio;
        }
//        nearestNeighbours.clear();
        classifyInstance(test);

        
        for (Instance instance : nearestNeighbours) {
            int classValue = (int) instance.classValue();
            distRatio[classValue]++;
        }

        for (int i = 0; i < distRatio.length; i++) {
            double x = distRatio[i];
            distRatio[i] = ((x / nearestNeighbours.size()));
        }
//        System.out.println(Arrays.toString(distRatio));
        return distRatio;
    }
}
