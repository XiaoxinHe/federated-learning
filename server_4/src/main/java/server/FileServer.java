package server;

import org.nd4j.linalg.api.ndarray.INDArray;
import play.mvc.WebSocket;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.*;

public class FileServer {

    public static String onDeviceModelPath = "res/onDeviceModel";
    public static Map<Integer, Map<String, INDArray>>cache= new HashMap<>();
    public static FederatedModel federatedmodel=new FederatedModel();
    private static ServerSocket serverSocket;
    public static Set<Integer>selected_clients=new HashSet<>();

    private void init(int port, int timeout) {
        try {
            serverSocket = new ServerSocket(port);
            serverSocket.setSoTimeout(timeout);
        } catch (Exception e) {
            System.out.println("Couldn't open socket. " + e.getMessage());
            return;
        }
        System.out.println("Server started on port " + port + " with timeout " + timeout + "ms");
    }

    private void run() {

        Integer curID = 1;

        Iterator iter=selected_clients.iterator();

        //noinspection InfiniteLoopStatement
//        while (!checkStopCondition()) {
        while (iter.hasNext()) {
            try {
                Socket clientSocket = serverSocket.accept();
                new Thread((Runnable) new ServerConnection(clientSocket, (Integer) iter.next())).start();
                System.out.println("client " + curID + " connected!");
                curID++;
            } catch (IOException e) {
                System.out.println("Error accepting client connection: " + e.getMessage());
            }
        }
    }

    public static boolean checkStopCondition() {

        int numberOfClient = 100;
        System.out.println("FileServer.cache.size(): "+FileServer.cache.size());
        return FileServer.cache.size() >= numberOfClient;

    }

    private static int getRandomNumberInRange(int min, int max) {

        if (min >= max) {
            throw new IllegalArgumentException("max must be greater than min");
        }

        Random r = new Random();
        return r.nextInt((max - min) + 1) + min;
    }

    public static void random_select(int C){
        while(selected_clients.size()<C){
            Integer tmp;
            tmp=getRandomNumberInRange(1,100);
            selected_clients.add(tmp);
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        // Run server
        int DEFAULT_PORT = 8000;
        int DEFAULT_TIMEOUT = 120 * 1000;//30 seconds
        int C=10;


        //to do: select client
        federatedmodel.initModel();

        FileServer fileserver = new FileServer();
        fileserver.init(DEFAULT_PORT, DEFAULT_TIMEOUT);

        for (int r = 0; r < 5; r++) {
            System.out.println("\n\nround:" + r);
            selected_clients.clear();
            random_select(C);
            fileserver.run();
            Thread.sleep(30*1000);
            federatedmodel.AverageWeights(2, 0.5, cache.size());
        }

    }

}
