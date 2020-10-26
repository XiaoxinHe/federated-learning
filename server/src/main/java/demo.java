import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class demo {

    public static int numInputs = 45;
    public static int numOutputs = 19;
    public static int numOfClients = 2374;
    public static int batchSize = 10;
    public static int nEpochs = 10;
    public static Map<Integer, Map<String, INDArray>> table = new HashMap<>();

    private static final Set<Integer> selected = new HashSet<>();

    private static MultiLayerNetwork model = null;
    private static MultiLayerNetwork transferred_model = null;
    private static final String serverModel = "res/model/demo.zip";

    public static void initModel() {
        int seed = 100;
        double learningRate = 0.001;
        int numHiddenNodes = 1000;

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
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        model = new MultiLayerNetwork(conf);

        model.init();
        System.out.println("init model finish");
    }

    public static void DeviceReg(int numbersNeeded) {
        if (numOfClients < numbersNeeded) {
            throw new IllegalArgumentException("Can't ask for more numbers than are available");
        }
        Random rng = new Random();
        while (selected.size() < numbersNeeded) {
            Integer next = rng.nextInt(numOfClients) + 1;
            selected.add(next);
        }
    }

    public static void ModelDelivery() {
        //Set up a fine-tune configuration
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(100)
                .build();
        //transfer model
        transferred_model = new TransferLearning.Builder(model)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(1)
                .build();
    }

    public static Map<String, INDArray> LocalTraining(int ID) {
        System.out.println("Hello from client " + ID);
        MultiLayerNetwork local_model = transferred_model;
        // load data
        RecordReader rr = new CSVRecordReader();
        String filename = "/Users/cautious/PycharmProjects/pythonProject/out/client/" + ID + ".csv";
        try {
            rr.initialize(new FileSplit(new File(filename)));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }

        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 19);
        System.out.println("load data finish");

        local_model.fit(trainIter, nEpochs);
        System.out.println("fit model finish");

        //write paramTable
        return local_model.paramTable();
    }

    public static void GlobalAgg() {
        int layer = 2;
        double alpha = 0.5;

        //original model
        Map<String, INDArray> paramTable = model.paramTable();
        INDArray weight = paramTable.get(String.format("%d_W", layer));
        INDArray bias = paramTable.get(String.format("%d_b", layer));

        INDArray avgWeights = weight.mul(alpha);
        INDArray avgBias = bias.mul(alpha);

        for (Map.Entry<Integer, Map<String, INDArray>> entry : table.entrySet()){
            weight = entry.getValue().get(String.format("%d_W", layer));
            bias = entry.getValue().get(String.format("%d_b", layer));
            avgWeights = avgWeights.add(weight.mul(1.0 - alpha).div(numOfClients));
            avgBias = avgBias.add(bias.mul(1.0 - alpha).div(numOfClients));
        }

        model.setParam(String.format("%d_W", layer), avgWeights);
        model.setParam(String.format("%d_b", layer), avgBias);

        selected.clear();
        table.clear();
    }

    public static void Evaluate() throws IOException {
        String filenameTest = "res/dataset/test.csv";

        //Load the test data:
        RecordReader rrTest = new CSVRecordReader();
        try {
            rrTest.initialize(new FileSplit(new File(filenameTest)));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }

        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 19);
        Evaluation eval = new Evaluation(numOutputs);
        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);
            eval.eval(labels, predicted);
        }

        // Print the evaluation statistics
        System.out.println(eval.stats());
        File file = new File("res/out/demo.txt");
        FileWriter fr = new FileWriter(file, true);
        fr.write(eval.stats());
        fr.close();
    }

    public static void main(String[] args) throws IOException {
        initModel();
        for (int i = 0; i < 1000; i++) {
            DeviceReg(100);
            ModelDelivery();
            for (Integer integer : selected) {
                table.put(integer, LocalTraining(integer));
            }
            GlobalAgg();
            Evaluate();

            ModelSerializer.writeModel(model, serverModel, false);
        }
    }

}

