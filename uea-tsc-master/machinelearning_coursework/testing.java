/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning_coursework;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Random;
import static machinelearning_coursework.MachineLearning_Coursework.writeResult;
import static machinelearning_coursework.MachineLearning_Coursework.writeResultEnsemble;
import static machinelearning_coursework.WekaTools.confusionMatrix;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.PropertyPath.Path;

/**
 *
 * @author yqm15fqu
 */
public class testing {

    public static void startTest(Instances train, Instances test, String probName, int foldNo) throws FileNotFoundException, Exception {
        String filePath = "\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\";
        String x = probName.substring(0, probName.length() - 5);
        KNN kNN = new KNN();

        kNN.setStandardise(false);
        StringBuilder result = new StringBuilder();

        if (!new File(filePath + "1NN").exists()) {
            new File(filePath + "1NN").mkdirs();
        }

        if (!new File(filePath + "1NN\\" + probName + "").exists()) {
            new File(filePath + "1NN\\" + probName + "").mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new File(filePath + "\\1NN\\" + x + "\\testFold" + foldNo + ".csv"))) {
            kNN.buildClassifier(train);
            result.append(probName).append(", 1NN").append('\n');
            writeResult(test, kNN, result, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

        public static void startTest1(Instances train, Instances test, String probName, int foldNo) throws FileNotFoundException, Exception {
        String filePath = "\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\";
        String x = probName.substring(0, probName.length() - 5);
        KNN kNN = new KNN();

        kNN.setStandardise(true);
        StringBuilder result = new StringBuilder();

        if (!new File(filePath + "Standardise").exists()) {
            new File(filePath + "Standardise").mkdirs();
        }

        if (!new File(filePath + "Standardise\\" + probName + "").exists()) {
            new File(filePath + "Standardise\\" + probName + "").mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new File(filePath + "\\Standardise\\" + x + "\\testFold" + foldNo + ".csv"))) {
            kNN.buildClassifier(train);
            result.append(probName).append(", Standardise").append('\n');
            writeResult(test, kNN, result, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
            public static void startTest2(Instances train, Instances test, String probName, int foldNo) throws FileNotFoundException, Exception {
        String filePath = "\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\";
        String x = probName.substring(0, probName.length() - 5);
        KNN kNN = new KNN();

        kNN.setStandardise(false);
        kNN.setLeaveOneOut(true);
        StringBuilder result = new StringBuilder();

        if (!new File(filePath + "LOOCV").exists()) {
            new File(filePath + "LOOCV").mkdirs();
        }

        if (!new File(filePath + "LOOCV\\" + probName + "").exists()) {
            new File(filePath + "LOOCV\\" + probName + "").mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new File(filePath + "\\LOOCV\\" + x + "\\testFold" + foldNo + ".csv"))) {
            kNN.buildClassifier(train);
            result.append(probName).append(", LOOCV").append('\n');
            writeResult(test, kNN, result, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }    public static void startTest3(Instances train, Instances test, String probName, int foldNo) throws FileNotFoundException, Exception {
        String filePath = "\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\";
        String x = probName.substring(0, probName.length() - 5);
        KNN kNN = new KNN();

        kNN.setStandardise(false);
        kNN.setWeightedVoting(true);
        StringBuilder result = new StringBuilder();

        if (!new File(filePath + "Weighted").exists()) {
            new File(filePath + "Weighted").mkdirs();
        }

        if (!new File(filePath + "Weighted\\" + probName + "").exists()) {
            new File(filePath + "Weighted\\" + probName + "").mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new File(filePath + "\\Weighted\\" + x + "\\testFold" + foldNo + ".csv"))) {
            kNN.buildClassifier(train);
            result.append(probName).append(", Weighted").append('\n');
            writeResult(test, kNN, result, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
        public static void startEnsemble(Instances train, Instances test, String probName, int foldNo)throws FileNotFoundException, Exception {
//        String filePath = "\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\";
        String filePath = "C:\\Users\\Rob\\Desktop\\Machine Learning Cw\\machinelearning\\testing\\Ensemble\\";
        String x = probName.substring(0, probName.length() - 5);
        KnnEnsemble kNN = new KnnEnsemble();

        StringBuilder result = new StringBuilder();

        if (!new File(filePath + probName + "").exists()) {
            new File(filePath + probName + "").mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new File(filePath + probName + "\\Ensemble" + x + "Fold" + foldNo + ".csv"))) {
            kNN.buildEnsemble(train);
            result.append(probName).append(", Ensemble").append('\n');
            writeResultEnsemble(test, kNN, result, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
            
    public static void startTest5(Instances train, Instances test, String probName, int foldNo) throws FileNotFoundException, Exception {
        String filePath = "\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\Results\\NaiveBayes\\Predictions\\";
        String x = probName.substring(0, probName.length() - 5);
        NaiveBayes nBayes = new NaiveBayes();
        StringBuilder result = new StringBuilder();

        if (!new File(filePath + probName + "").exists()) {
            new File(filePath + x + "").mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new File(filePath + x + "\\" + "testFold" + foldNo + ".csv"))) {
            nBayes.buildClassifier(train);
            result.append(probName).append(", naiveBayes").append('\n');
            writeResult(test, nBayes, result, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void startTest6(Instances train, Instances test, String probName, int foldNo) throws FileNotFoundException, Exception {
        String filePath = "\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\Results\\RandomForest\\Predictions\\";
        String x = probName.substring(0, probName.length() - 5);
        Random rand = new Random();
        RandomForest randFor = new RandomForest();
        randFor.setSeed(rand.nextInt());
        StringBuilder result = new StringBuilder();

        if (!new File(filePath + probName + "").exists()) {
            new File(filePath + x + "").mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new File(filePath + x + "\\" + "testFold" + foldNo + ".csv"))) {
            randFor.buildClassifier(train);
            result.append(probName).append(", RandomForest").append('\n');
            writeResult(test, randFor, result, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void startTest7(Instances train, Instances test, String probName, int foldNo) throws FileNotFoundException, Exception {
        String filePath = "\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\Results\\RotationForest\\Predictions\\";
        String x = probName.substring(0, probName.length() - 5);
        RotationForest rotFor = new RotationForest();
        StringBuilder result = new StringBuilder();

        if (!new File(filePath + probName + "").exists()) {
            new File(filePath + x + "").mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new File(filePath + x + "\\" + "testFold" + foldNo + ".csv"))) {
            rotFor.buildClassifier(train);
            result.append(probName).append(", RotationForest").append('\n');
            writeResult(test, rotFor, result, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void findStats(int[] actual, int[] predicted) {

        int[] act = {1};
        int[] pre = {1};

        int[][] answer = confusionMatrix(predicted, actual);

        for (int i = 0; i < answer.length; i++) {
            for (int j = 0; j < answer.length; j++) {
                System.out.println(answer[i][j] + "\t");
            }
            System.out.println("\n");
        }
        System.out.println(Arrays.deepToString(answer));

        double a = answer[0][0];
        double b = answer[0][1];
        double c = answer[1][0];
        double d = answer[1][1];

        double TPR = a / (a + c);
        double FPR = b / (b + d);
        double FNR = c / (a + c);
        double TNR = d / (b + d);

        System.out.println("TPR: " + TPR + " FPR: " + FPR + " FNR: " + FNR + " TNR: " + TNR);
    }

}
