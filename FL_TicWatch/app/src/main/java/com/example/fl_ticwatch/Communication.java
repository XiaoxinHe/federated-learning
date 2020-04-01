package com.example.fl_ticwatch;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.util.Map;

public class Communication {

    // Connection details
    public int id;

    private Socket socket;
    private DataInputStream in;
    private DataOutputStream out;

    private Communication(Socket socket, int id, DataInputStream input, DataOutputStream output) {
        this.socket = socket;
        this.id = id;
        this.in = input;
        this.out = output;
        System.out.println("id:"+id);
    }

    // Factory method to create a client instance
    public static Communication connect(String ip, int port, int timeout) {
        try {
            System.out.println("Connecting to server...");
            Socket socket = new Socket(ip, port);
            socket.setSoTimeout(timeout);
            DataInputStream in = new DataInputStream(socket.getInputStream());
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            System.out.println("Connected:"+socket.isConnected());
            int id = in.readInt();
            System.out.println("id:"+id);
            return new Communication(socket, id, in, out);
        } catch (IOException e) {
            // Handle errors
            System.out.println(e.getMessage());
            return null;
        }
    }

    // Attempts to quit gracefully using operations
    public void quit() {
        try {
            out.writeUTF("QUIT");
            out.close();
            in.close();
            socket.close();
        } catch (IOException e) {
            System.out.println("Error quitting gracefully (" + e.getMessage() + ")");
            System.out.println("Force closing");
            // Force close
            try {
                out.close();
            } catch (IOException f) { /* Do nothing */ }
            try {
                in.close();
            } catch (IOException f) { /* Do nothing */ }
            try {
                socket.close();
            } catch (IOException f) { /* Do nothing */ }

        }
        System.out.println("Session closed");
    }

    private byte[] downloadFromServer(String filename) throws IOException {
        // Send operation and filename
        System.out.println("Sending DWLD operation to server...");
        out.writeUTF("DWLD");
        out.writeShort(filename.length());
        out.writeChars(filename);

//         Read server response, handle weird values (out of spec)
        int fileSize = in.readInt();
        if (fileSize == -1) {
            System.out.println("File does not exist on server");
            return null;
        } else if (fileSize < 0) {
            System.out.println("Negative integer returned for filesize that was not -1. Download cancelled");
            return null;
        }

        // Confirm readiness to download
        out.writeBoolean(true);
        System.out.println("Downloading from server...");

        // Declare our array of bytes
        byte[] bytes = new byte[fileSize];
        int totBytesRead = 0;

        // Read as many bytes as possible until buffer is full
        while (totBytesRead < fileSize) {
            int bytesRead = in.read(bytes, totBytesRead, fileSize - totBytesRead);
            totBytesRead += bytesRead;
        }

        return bytes;
    }


    public boolean download(String filename) throws IOException {
        // Start timer
        long startTime = System.currentTimeMillis();

        // Download bytes from server
        byte[] bytes;
        try {
            bytes = downloadFromServer(filename);
        } catch (IOException e) {
            // Handle errors, errors here should cause a disconnect
            e.printStackTrace();
            return false;
        }

        if (bytes != null) {
            // Write file out
            File outFile = ClientModel.locateToLoadModel;
            FileOutputStream stream = new FileOutputStream(outFile);
            stream.write(bytes);

            // Gather statistics
            long endTime = System.currentTimeMillis();
            double timeTaken = (endTime - startTime);
            timeTaken /= 1000;
            System.out.println(String.format("%,d bytes transferred in %,.4fs", bytes.length, timeTaken));
        }
        return true;
    }

    public void uploadParamTable(Map<String, INDArray> paramTable) throws IOException {
        System.out.println("Sending UPLDPT operation to server and waiting for response...");
        out.writeUTF("UPLDPT");

        // Get server confirmation
        if (!in.readBoolean()) {
            System.out.println("Server rejected request:uploadParamTable");
            return;
        }

        // Convert Map to byte array
        System.out.println("uploading ParamTable...");
        ObjectOutputStream mapOutputStream = new ObjectOutputStream(out);
        mapOutputStream.writeObject(paramTable);
        System.out.println("upload ParamTable finish!");
    }

}
