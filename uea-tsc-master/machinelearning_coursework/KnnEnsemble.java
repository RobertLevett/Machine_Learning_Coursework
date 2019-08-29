/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning_coursework;

import static java.lang.Math.floor;
import java.util.ArrayList;
import machinelearning_coursework.*;
import static machinelearning_coursework.KNN.mode;
import weka.classifiers.Classifier;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author yqm15fqu
 */
public class KnnEnsemble {

    private int ensembleSize = 50;
    private ArrayList<KNN> ensembleContainer = new ArrayList(ensembleSize);
    private Instances trainData;
    private boolean removeRandAttributes = false;
    private double[] ensemblePredictions = new double[ensembleSize];
    // for removing attributes.
    public void setRemoveRandAtt(boolean x) {
        removeRandAttributes = x;
    }
    
    //distribution for instance for the ensemble.
    public double[] getEnsemblePredictions(){
        double [] classVals = new double[trainData.numClasses()];
        for (int i = 0; i < ensemblePredictions.length; i++) {
            classVals[(int) ensemblePredictions[i]]++;
        }
        for (int i = 0; i < classVals.length; i++) {
            classVals[i] = (classVals[i]/ensemblePredictions.length);
        }
        return classVals;
    }

    public void setEndembleSize(int x) {
        this.ensembleSize = x;
        ensemblePredictions = new double[ensembleSize];
    }

    public int getEnsembleSize() {
        return ensembleSize;
    }
    //calculate ensemble accuracy by looking at what each classifier picks the 
    //class value as, make a distribution, pick the highest.
    public static double accuracyEns(KnnEnsemble c, Instances test) throws Exception {

        double result = 0;
        double accuracy = 0;
        int counter = 0;
        for (Instance instance : test) {
            result = c.classifyInstanceEns(instance);
            if (result == instance.classValue()) {
                counter++;
            }
        }
        accuracy = (((double) counter / test.size()) * 100);

        return accuracy;
    }

    public void buildEnsemble(Instances ins) throws Exception {
        Random rand = new Random();

        //So that the Data is randomised
        ins.randomize(rand);
        Instances ensembleInstances = new Instances(ins, 0);
        int sampleSize = (int) Math.floor(ins.size() * 0.3);
        int attributeSize = (int) Math.floor(ins.numAttributes() * 0.4);

        if (removeRandAttributes) {
            int[] indices = new int[attributeSize];

            for (int i = 0; i < 10; i++) {
                int a = ins.numAttributes() - 1;
                for (int k = 0; k < attributeSize - 1; k++) {
                    indices[k] = rand.nextInt(a);
                    a--;
                }
                Remove removeFilter = new Remove();
                removeFilter.setAttributeIndicesArray(indices);
                removeFilter.setInputFormat(ins);
                trainData = Filter.useFilter(ins, removeFilter);
            }
        } else {
            trainData = ins;
        }
        //populate ensemble
        for (int i = 0; i < ensembleSize; i++) {
            KNN kNN = new KNN();
            Instances tempInst = ins;
            tempInst.randomize(rand);

            for (int j = 0; j < sampleSize; j++) {
//                Instance inst = trainData.get(x).deleteAttributeAt(i);
                ensembleInstances.add(tempInst.get(j));
            }
            kNN.setLeaveOneOut(true);
            kNN.setStandardise(true);
            kNN.setWeightedVoting(true);
            kNN.buildClassifier(ensembleInstances);
            ensembleInstances = new Instances(trainData, 0);
//            System.out.println("KNN: " + i + " added. \n \n");
            ensembleContainer.add(kNN);
        }
        System.out.println("Completed Ensemble build.");
    }

    public double classifyInstanceEns(Instance test) {
        
        for (int i = 0; i < ensembleContainer.size(); i++) {
            ensemblePredictions[i] = ensembleContainer.get(i).classifyInstance(test);
        }

//        System.out.println("THIS IS THE ENSEMBLE   ANSWER: " + mode(ensemblePredictions));
        double Answer = mode(ensemblePredictions);

        return Answer;
    }

}
