package server;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.MathFunction;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Random;

public class baseline_2 {

    public static final int seed = 12345;
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 10;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int batchSize = 100;

    //Network learning rate
    public static final double learningRate = 0.01;
    public static final int numInputs = 19;
    private static final int numOutputs = 1;

    private static MultiLayerConfiguration getDeepDenseLayerNetworkConfiguration() {
        final int numHiddenNodes = 50;
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

//                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
//                        .activation(Activation.RELU)
//                        .build())
//                .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
//                        .activation(Activation.RELU)
//                        .build())
//                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .activation(Activation.IDENTITY) //linear
//                        .nIn(numHiddenNodes).nOut(numOutputs).build())
//                .build();
    }


    public static void main(String[] args) throws Exception {

        //Switch these two options to do different functions with different networks
        final MultiLayerConfiguration conf = getDeepDenseLayerNetworkConfiguration();

        final String filenameTrain  = "res/dataset_2/train.csv";
        final String filenameTest  = "res/dataset_2/test.csv";

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,numOutputs);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,numOutputs);
        System.out.println("data finish");


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        System.out.println("config finish");

        model.init();
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates
        System.out.println("init finish");


        model.fit( trainIter, nEpochs );
        System.out.println("fit finish");
//        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("res/model_2/trained_nn.zip");



        RegressionEvaluation eval = model.evaluateRegression(testIter);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features,false);
            System.out.println("label:");
            System.out.println(labels);

            System.out.println("predicted:");
            System.out.println(predicted);
            System.out.println("\n\n");

        }

        // Print the evaluation statistics
        System.out.println(eval.stats());

        // save model
        boolean saveUpdate = true;
        File locationToSave = new File("res/model/trained_nn_2.zip");
        ModelSerializer.writeModel(model, locationToSave, saveUpdate);
        System.out.println("save model finish");
    }

}
