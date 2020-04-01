package com.example.fl_ticwatch;

import android.os.Environment;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.io.File;
import java.io.IOException;

public class ClientModel {

    private static final File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
    static final File locateToSaveDataSet = new File(dir, "labelled_dataset.csv");
    static final File locateToLoadModel = new File(dir, "trained_nn.zip");
    static String id = null;

    private static final int numHiddenNodes = 1000;
    private static final int numOutputs = 10;
    private static final int nEpochs = 10;
    private static final int batchSize = 10;


    public static MultiLayerNetwork model = null;
    public static MultiLayerNetwork transferred_model = null;

    //Set up a fine-tune configuration
    public static FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(5e-5))
            .seed(100)
            .build();

    public static void TrainingModel(File file) throws IOException {

        //load model
        model = ModelSerializer.restoreMultiLayerNetwork(locateToLoadModel, false);

        //transfer model
        transferred_model = new TransferLearning.Builder(model)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(1)
                .build();

        System.out.println("loading data...");
        RecordReader rr = new CSVRecordReader();
        try {
            rr.initialize(new FileSplit(file));
        } catch (Exception e) {
            e.printStackTrace();
        }

        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, 10, 0, 10);

        System.out.println("load data finish");

        System.out.println(model);

        transferred_model.fit(trainIter, nEpochs);

        //
        rr.close();

    }
}