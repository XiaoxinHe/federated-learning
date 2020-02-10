import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;

public class FileServer {

    public static String onDeviceModelPath = "res/onDeviceModel";
    private static ServerSocket serverSocket;

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

        int curID = 1;

        //noinspection InfiniteLoopStatement
        while (!checkStopCondition()) {

            try {
                Socket clientSocket = serverSocket.accept();
//                clientSocket.setSoTimeout(timeout);
                new Thread((Runnable) new ServerConnection(clientSocket, curID)).start();
                curID++;
            } catch (IOException e) {
                System.out.println("Error accepting client connection: " + e.getMessage());
            }
        }

    }

    public static boolean checkStopCondition() {

        int numberOfClient = 10;

        File dir = new File(onDeviceModelPath);
        File[] listOfFiles = dir.listFiles();

        return listOfFiles.length >= numberOfClient;

    }


    public static void main(String[] args) throws IOException {
        // Run server
        int DEFAULT_PORT = 1234;
        int DEFAULT_TIMEOUT = 30 * 1000;//30 seconds


        FederatedModel federatedmodel = new FederatedModel();
        federatedmodel.initModel();

        FileServer fileserver = new FileServer();
        fileserver.init(DEFAULT_PORT, DEFAULT_TIMEOUT);

        for (int r = 0; r < 5; r++) {
            System.out.println("\n\nround:" + r);
            fileserver.run();
            federatedmodel.AverageWeights(2, 0.5);
        }

    }

}
