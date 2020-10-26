import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

//import org.deeplearning4j.api.storage.StatsStorage;
//import org.deeplearning4j.ui.storage.FileStatsStorage;
//import org.deeplearning4j.ui.api.UIServer;
//import org.deeplearning4j.ui.stats.StatsListener;


public class baseline_centralized {

    public static void main(String[] args) throws Exception {

        int seed = 100;
        double learningRate = 0.001;
        int batchSize = 20;
        int nEpochs = 2;

        int numInputs = 45;
        int numOutputs = 19;
        int numHiddenNodes = 1000;

        final String filenameTrain  = "res/dataset/train.csv";
        final String filenameTest  = "res/dataset/test.csv";

        // Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,19);

        // Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,19);
        System.out.println("data finish");

        // https://deeplearning4j.org/docs/latest/deeplearning4j-nn-multilayernetwork
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX) //linear
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        System.out.println("config finish");

        model.init();
        //model.setListeners(new ScoreIterationListener(100));  //Print score every 10 parameter updates
        System.out.println("init finish");

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        // DL4J UI Server started at http://localhost:9000

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        model.setListeners(new StatsListener(statsStorage));

        model.fit( trainIter, nEpochs );
        System.out.println("fit finish");

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features,false);
            eval.eval(labels, predicted);
        }
        // Print the evaluation statistics
        System.out.println(eval.stats());

        // Save model
        boolean saveUpdate = true;
        File locationToSave = new File("res/model/trained_nn.zip");
        ModelSerializer.writeModel(model, locationToSave, saveUpdate);
        System.out.println("save model finish");
    }

}

/*
========================Evaluation Metrics========================
 # of classes:    19
 Accuracy:        0.9857
 Precision:       0.9860
 Recall:          0.9857
 F1 Score:        0.9857
Precision, recall & F1: macro-averaged (equally weighted avg. of 19 classes)


=========================Confusion Matrix=========================
    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18
------------------------------------------------------------------------------------------------
 7428    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0 | 0 = 0
    0 7408    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    3    0 | 1 = 1
    0    0 7266    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0 | 2 = 2
    0    0    0 7317    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0 | 3 = 3
    0    2    0    0 7340    0    0    2    0    3    1    0    0    0    0    0    0    2    5 | 4 = 4
    0    0    0    0  383 6926   13   23    2    0    0    0    0    0    0    0    0   13    4 | 5 = 5
    0    0    0    0    0    0 7427   33    0    0    0    0    0    0    0    0    0    0    2 | 6 = 6
    0   11    0    0   68   61  277 6816    7    0    0    0    2    1    1    1    0    7   91 | 7 = 7
    0    0    0    0    0    0    0    1 7352   61   25    0    0    0    0    0    0    1   12 | 8 = 8
    0    0    0    0    0    0    0    0    1 7304   94    1    1    1    0    0    0    0    6 | 9 = 9
    0    0    0    0    0    0    0    0    1   34 7358    0    4    5    0    1    0    0    1 | 10 = 10
    0    0    0    0    0    0    0    0    0    6    2 7200    0    0    0    0    0    1   12 | 11 = 11
    0    0    0    0    0    0    0    0    1    0    0    1 7400    7    0    0    0    0   19 | 12 = 12
    0    0    0    0    0    0    0    0    0    1    0    0    2 7197    0    0    0    0    5 | 13 = 13
    1    0    0    0    0    0    0    0    0    0    0    0    0    0 7453    1    0    0    0 | 14 = 14
    0    0    0    0    0    0    0    0    0    0    6    0    0    1    0 7276    0    0    5 | 15 = 15
    0    0    1    0    0    0    0    0    0    0    0    0    0    0    0    0 7362    0    0 | 16 = 16
    0    4    0    0  124   35    0   12    8    1    2    4    0    0    0    1    0 7214   27 | 17 = 17
    0    2    0    0   14    7    0   50   18   54   40   65   54   52    0    9    1   79 6953 | 18 = 18

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
 */