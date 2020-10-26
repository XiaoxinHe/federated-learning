import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
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
import java.util.HashMap;
import java.util.Map;

public class baseline_distributed {
    public static int numInputs = 45;
    public static int numOutputs = 19;
    public static int batchSize = 4000;
    public static int numOfClients = 2375; //2375;
    public static int rounds = 20;

    public static MultiLayerNetwork model = null;
    public static Map<Integer, Map<String, INDArray>> cache = new HashMap<>();

    private static final String serverModel = "res/model/server_model.zip";

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
        System.out.println("config finish");

        model.init();
        System.out.println("init model finish");
    }

    public static void modelDelivery() throws IOException {
        ModelSerializer.writeModel(model, serverModel, true);
    }

    public static void FedAvg() throws IOException {
        int layer = 2;
        double alpha = 0.5;

        //original model
        Map<String, INDArray> paramTable = model.paramTable();
        INDArray weight = paramTable.get(String.format("%d_W", layer));
        INDArray bias = paramTable.get(String.format("%d_b", layer));

        INDArray avgWeights = weight.mul(alpha);
        INDArray avgBias = bias.mul(alpha);

        for (int i = 0; i < numOfClients; i++) {
            paramTable = cache.get(i);
            weight = paramTable.get(String.format("%d_W", layer));
            bias = paramTable.get(String.format("%d_b", layer));
            avgWeights = avgWeights.add(weight.mul(1.0 - alpha).div(numOfClients));
            avgBias = avgBias.add(bias.mul(1.0 - alpha).div(numOfClients));
        }

        model.setParam(String.format("%d_W", layer), avgWeights);
        model.setParam(String.format("%d_b", layer), avgBias);

        System.out.println("\nWriting server model...");
        ModelSerializer.writeModel(model, serverModel, false);

        cache.clear();
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
        File file = new File("res/out/xxx.txt");
        FileWriter fr = new FileWriter(file, true);
        fr.write(eval.stats());
        fr.close();
    }

    public static void main(String[] args) throws Exception {
        initModel();
        Client client = new Client();
        for (int i = 0; i < rounds; i++) {
            System.out.println("Round " + i);
            Client.TransferModel(model);
            for (int j = 0; j < numOfClients; j++) {
                client.setID(j);
                client.LocalTraining();
                cache.put(j, client.paramTable);
            }
            FedAvg();
            Evaluate();
        }
        System.out.println("Done!");
    }

}
