package server;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.file.Files;
import java.util.Map;

public class ServerConnection implements Runnable {
    private int id;
    private Socket socket;
    private DataInputStream input;
    private DataOutputStream output;

    ServerConnection(Socket clientSocket, int id) {
        this.socket = clientSocket;
        this.id = id;
    }

    public void run() {
        try {
            input = new DataInputStream(socket.getInputStream());
            output = new DataOutputStream(socket.getOutputStream());

            //send id
            output.writeInt(id);

            mainLoop();

            output.close();
            input.close();
            socket.close();
        } catch (IOException | ClientError e) {
            System.out.println("Input/Output error occurred: " + e.getMessage());
            System.out.println("Closing client connection forcefully");

            try { output.close(); } catch (IOException f) { /* Do nothing */ }
            try { input.close(); } catch (IOException f) { /* Do nothing */ }
            try { socket.close(); } catch (IOException f) { /* Do nothing */ }
        }
    }

    private void mainLoop() throws IOException, ClientError {
        wait:
        while (true) {
            String operation;
            try {
                operation = input.readUTF();
            } catch (SocketTimeoutException e) {
                // Socket will continually timeout as no input is received unless prompted by the client
                continue;
            }

            switch (operation) {
                case "UPLD"://client upload file to server
                    upload();
                    break;
                case "UPLDPT"://client upload parametertable to server
                    try {
                        uploadParametertable();
                    } catch (ClassNotFoundException e) {
                        e.printStackTrace();
                    }
                    break;
                case "DWLD"://client download file from server
                    download();
                    break;
                case "QUIT":
                    System.out.println("QUIT triggered by client\n");
                    break wait;
                default:
                    System.out.println("Operation unknown: " + operation);
                    System.out.println("Terminating connection due to client error");
                    break wait;
            }
        }
    }

    private void uploadParametertable() throws IOException, ClassNotFoundException {

        // Start timer
        System.out.println("Ready to receive parameter table");//t1
        output.writeBoolean(true);

//        long t3 = System.currentTimeMillis();
        long t3=System.nanoTime();

        ObjectInputStream mapInputStream = new ObjectInputStream(input);
//        long ttemp=System.nanoTime();
        Map<String, INDArray> paramTable = (Map) mapInputStream.readObject();

//        long t4 = System.currentTimeMillis();
        long t4 = System.nanoTime();

        double timeTaken = (t4 - t3);
        timeTaken/=1000;

        String response = String.format("bytes transferred in %,.3f us", timeTaken);
//        System.out.println(ttemp-t3);
        output.writeDouble(timeTaken);
        System.out.println(response);
        FileServer.cache.put(id, paramTable);
    }

    private void upload() throws IOException, ClientError {

        System.out.println("Client is requesting to upload a file");

        // Start timer
        long startTime = System.currentTimeMillis();

        String fileName = getFilename(true);
        String fullPath = filenameAddBaseDir(fileName);
        System.out.println("Filename: " + fileName);

        // Get filesize
        int fileSize = input.readInt();
        if (fileSize < 0) {
            throw new ClientError("File size is less than 0 (" + fileSize + ")", true);
        }
        System.out.println("Filesize: " + fileSize);

        // Receive data from client
        System.out.println("Ready to receive data");
        output.writeBoolean(true);

        // Declare our array of bytes
        byte[] bytes = new byte[fileSize];
        int totBytesRead = 0;

        // Read as many bytes as possible until buffer is full
        while (totBytesRead < fileSize) {
            int bytesRead = input.read(bytes, totBytesRead, fileSize - totBytesRead);
            totBytesRead += bytesRead;
        }

        // Write file out
        File outFile = new File(fullPath);
        //noinspection ResultOfMethodCallIgnored
        try (FileOutputStream stream = new FileOutputStream(outFile)) {
            stream.write(bytes);
        } catch (IOException e) {
            System.out.println("Error writing file to disk");
            System.out.println(e.getMessage());
            output.writeUTF("Server error, could not write to disk (" + e.getMessage() + ")");
        }

        // Gather statistics
        long endTime = System.currentTimeMillis();
        double timeTaken = (endTime - startTime);
        timeTaken /= 1000;
        String response = String.format("%,d bytes transferred in %,.2fs", fileSize, timeTaken);

        System.out.println(response);
        output.writeUTF(response);
        System.out.println("Upload finished");

    }


    private void download() throws IOException, ClientError {

        System.out.println("Client is requesting to download a file");

        String filename = getFilename(false);
        String fullPath = "res/model/"+filename;

        // Check if file exists
        File file = new File(fullPath);
        if (!file.exists()) {
            System.out.println("The file \"" + filename + "\" does not exist on the server");
            output.writeInt(-1);
            return;
        }

        // Send the file size back to the client
        // Since we're limited to 32 bit integers for the file size, then this will cause the server to crash on files larger than 2^31 bytes
        output.writeInt((int) file.length());

        // Read file from disk while we wait for client to respond
        System.out.println("Reading file from disk");
        byte[] bytes = Files.readAllBytes(file.toPath());

        // Wait for client to return ready
        if (!input.readBoolean()) {
            System.out.println("Client returned false for ready status");
            return;
        }

        // Send bytes to client
        output.write(bytes);
        System.out.println("Bytes sent");
    }



    private String getFilename(boolean sendErrorBack) throws IOException, ClientError {
        // Get length of filename
        short fileNameLen = input.readShort();
        if (fileNameLen < 1) {
            throw new ClientError("Length of filename was not a positive integer (received " + fileNameLen + ")", sendErrorBack);
        }

        // Read chars as filename
        char[] fileNameChar = new char[fileNameLen];
        for (int i = 0; i < fileNameLen; i++) {
            fileNameChar[i] = input.readChar();
        }

        return new String(fileNameChar);
    }

    private String filenameAddBaseDir(String filename) {
        return FileServer.onDeviceModelPath + "/" + filename;
    }

    class ClientError extends Exception {
        protected boolean sendErrorBack;

        ClientError(String message, boolean sendErrorBack) {
            super(message);
            this.sendErrorBack = sendErrorBack;
        }
    }
}
