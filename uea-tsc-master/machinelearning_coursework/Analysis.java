package machinelearning_coursework;

import java.io.File;
import java.io.IOException;
import evaluation.MultipleClassifierEvaluation;
import java.io.FileFilter;
import java.util.ArrayList;
import java.util.List;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author yqm15fqu
 */
public class Analysis {

    public static String[] findFoldersInDirectory(String directoryPath) {
        File directory = new File(directoryPath);

        FileFilter directoryFileFilter = new FileFilter() {
            public boolean accept(File file) {
                return file.isDirectory();
            }
        };

        File[] directoryListAsFile = directory.listFiles(directoryFileFilter);
        String[] foldersInDirectory = new String[directoryListAsFile.length];
        int counter = 0;
        for (File directoryAsFile : directoryListAsFile) {

            foldersInDirectory[counter] = directoryAsFile.getName();
            counter++;
        }

        return foldersInDirectory;
    }

    public static void makeDirs(String[] folders) {
//        String filePath = "\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\Results\\1NN\\";
        String filePath = " \\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\LAST REPO\\machinelearning\\testing\\Ensemble1\\";
       
        for (int i = 0; i < folders.length; i++) {
            new File(filePath + folders[i]).mkdirs();

        }

    }

    public static void changeNames(String[] folders) throws IOException {

        for (int j = 0; j < folders.length; j++) {

            File folder = new File("\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\Ensemble\\" + folders[j] + "\\");
            File[] listOfFiles = folder.listFiles();

            for (int i = 0; i < listOfFiles.length; i++) {

                if (listOfFiles[i].isFile()) {

                    File f = new File("\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\Ensemble\\" + folders[j] + "\\" + listOfFiles[i].getName());

                    f.renameTo(new File(folder + "\\" + "testFold" + i + ".csv"));
                }
            }
            System.out.println("conversion is done");
        }

    }

    public static void main(String[] args) throws Exception {

        String[] datasets = findFoldersInDirectory("\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\REAL TEST 1\\");

        String[] classifiers = findFoldersInDirectory("\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\Results\\");

        for (int i = 0; i < datasets.length; i++) {
            datasets[i] = (String) datasets[i].subSequence(0, datasets[i].length() - 5);
            System.out.print(datasets[i]+"\n");
        }
//        RenameFileDirectory(datasets);
//        makeDirs(datasets);
//
//        changeNames(datasets);
//
//        String[] datasets = {"1NNbank", "1NNblood", "1NNbreast-cancer-wisc-diag",
//            "1NNbreast-tissue", "1NN"};
////        String[] classifiers = new String[]{"1NN"};


        MultipleClassifierEvaluation mC = new MultipleClassifierEvaluation("\\\\ueahome4\\stusci1\\yqm15fqu\\data\\NTProfile\\Desktop\\temp test\\machinelearning\\testing\\Anlysis\\", "FourthAnal", 30);
        mC.setTestResultsOnly(true);
//            setTestResultsOnly(false).
        mC.setBuildMatlabDiagrams(true);
//            setBuildMatlabDiagrams(false).
        mC.setUseAccuracyOnly();

        mC.setDatasets(datasets);
        mC.readInClassifiers(classifiers, "//ueahome4/stusci1/yqm15fqu/data/NTProfile/Desktop/temp test/machinelearning/testing/Results/");
        mC.runComparison();
    }

}
