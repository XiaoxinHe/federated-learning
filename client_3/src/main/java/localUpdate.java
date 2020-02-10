import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.io.File;
import java.io.IOException;

public class localUpdate {

    String filenameTrain = "res/dataset/train";
    String id = null;



    private static final int nEpochs = 5;
    private static final int batchSize = 10;

    public static MultiLayerNetwork model = null;
    public static MultiLayerNetwork transferred_model = null;

    //Set up a fine-tune configuration
    public static FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(5e-5))
            .seed(100)
            .build();

    public void clientUpdate() {
        //load model from server
        System.out.println("loading model...");
        String inFile = FileClient.downloadDir+"server_model.zip";
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(inFile);
            System.out.println("load model finish!");
        } catch (IOException e) {
            e.printStackTrace();
        }

        //transfer model
        transferred_model = new TransferLearning.Builder(model)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(1)
                .build();

        System.out.println("loading data...");
        RecordReader rr = new CSVRecordReader();
        filenameTrain = filenameTrain + "_" + id + ".csv";
        try {
            rr.initialize(new FileSplit(new File(filenameTrain)));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }


        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 10);
        System.out.println("load data finish!");
        transferred_model.setListeners(new ScoreIterationListener(50));  //Print score every 10 parameter updates

        transferred_model.fit(trainIter, nEpochs);

        //write model
        String outFile = FileClient.uploadDir + id+".zip";
        try {
            ModelSerializer.writeModel(transferred_model, outFile, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void hello() {
        System.out.println("hello from localUpdate!");
        System.out.println(id);
    }

}
