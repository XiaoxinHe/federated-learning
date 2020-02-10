import java.io.*;
import java.net.Socket;
import java.nio.file.Files;

public class FileClient {


    // Connection details
    public int id;

    private Socket socket;
    private DataInputStream in;
    private DataOutputStream out;

    public static String uploadDir="res/model/";
    public static String downloadDir="res/onDeviceModel/";

    private FileClient(Socket socket, int id, DataInputStream input, DataOutputStream output) {
        this.socket = socket;
        this.id = id;
        this.in = input;
        this.out = output;
        System.out.println(id);
    }

    // Factory method to create a client instance
    public static FileClient connect(String ip, int port, int timeout) {
        try {
            System.out.println("Connecting to server...");
            Socket socket = new Socket(ip, port);
            socket.setSoTimeout(timeout);
            DataInputStream in = new DataInputStream(socket.getInputStream());
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());

            System.out.println("Connected");
            int id = in.readInt();
            return new FileClient(socket, id, in, out);
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

        // Read server response, handle weird values (out of spec)
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

    private boolean download(String filename) throws IOException {
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
            String localpath = downloadDir + filename;
            File outFile = new File(localpath);
            FileOutputStream stream = new FileOutputStream(outFile);
            stream.write(bytes);

            // Gather statistics
            long endTime = System.currentTimeMillis();
            double timeTaken = (endTime - startTime);
            timeTaken /= 1000;
            System.out.println(String.format("%,d bytes transferred in %,.2fs", bytes.length, timeTaken));
        }
        return true;
    }

    // Returns false if there is a SERVER error
    // Client errors (eg. IOException on file read, will still return true)
    public boolean upload(File file, String filename) {
        // Read the file as bytes from disk first
        System.out.println("Reading file from disk");
        byte[] bytes;
        try {
            bytes = Files.readAllBytes(file.toPath());
        } catch (IOException e) {
            // Handle errors. Errors reading file are not fatal to the server-client connection
            System.out.println(e.getMessage());
            e.printStackTrace();
            return true;
        }

        // Send the file to the server
        try {
            uploadFile(filename, bytes);
        } catch (IOException e) {
            // Handle errors
            System.out.println(e.getMessage());
            e.printStackTrace();
            return false;
        }

        return true;
    }

    // The code that performs the upload (wrapped in upload to handle errors)
    private void uploadFile(String filename, byte[] bytes) throws IOException {
        // Send operation, filename, and length of file
        System.out.println("Sending UPLD operation to server and waiting for response...");
        out.writeUTF("UPLD");
        out.writeShort(filename.length());
        out.writeChars(filename);
        out.writeInt(bytes.length);

        // Get server confirmation
        if (!in.readBoolean()) {
            String reason = in.readUTF();
            System.out.println("Server rejected request");
            System.out.println("Reason: " + reason);
            return;
        }

        // Send file
        System.out.println("Sending data to server...");
        out.write(bytes);
        System.out.println(in.readUTF());
    }


    public static void main(String args[]) throws IOException, InterruptedException {
        String DEFAULT_IP = "localhost";
        int DEFAULT_PORT = 1234;
        int DEFAULT_TIMEOUT = 5000;

        FileClient c = connect(DEFAULT_IP, DEFAULT_PORT, DEFAULT_TIMEOUT);

        //download latest model from server
        c.download("server_model.zip");

        //local update
        localUpdate localModel = new localUpdate();
        localModel.id = c.id + "";
        localModel.clientUpdate();

        //upload local model to server
        c.upload(new File(uploadDir + c.id + ".zip"), c.id + ".zip");

        //disconnect
        c.quit();

    }

}
