package com.example.fl_ticwatch;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.AsyncTask;
import android.os.BatteryManager;
import android.os.Bundle;
import android.provider.Settings;
import android.support.wearable.activity.WearableActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.core.app.ActivityCompat;

import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends WearableActivity {

    private TextView textView;
    private ProgressBar progressBar;

//    private AsyncTaskPull asyncTaskPull;
//    private AsyncTaskPush asyncTaskPush;
//    private AsyncTaskTrain asyncTaskTrain;
//    private AsyncTaskRunner runner;


    // read and write permissions
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };
    public static void verifyStoragePermission(Activity activity) {
        // Get permission status
        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission we request it
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
    }

    // access network state permissions
//    private static final int REQUEST_NETWORK_INFO = 1;
//    private static String[] PRMISSIONS_NETWORK={
//            Manifest.permission.ACCESS_NETWORK_STATE,
//            Manifest.permission.INTERNET
//    };
//    public static void verifyNetworkPermission(Activity activity) {
//        // Get permission status
//        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.ACCESS_NETWORK_STATE);
//        if (permission != PackageManager.PERMISSION_GRANTED) {
//            // We don't have permission we request it
//            ActivityCompat.requestPermissions(
//                    activity,
//                    PRMISSIONS_NETWORK,
//                    REQUEST_NETWORK_INFO
//            );
//        }
//    }

    private boolean isNetworkAvailable() {
        ConnectivityManager connectivityManager
                = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo activeNetworkInfo = null;
        if (connectivityManager != null) {
            activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
            System.out.println("activeNetworkInfo: "+activeNetworkInfo);
        }
        return activeNetworkInfo != null && activeNetworkInfo.isConnected();
    }

    // training related
    private static boolean isPulling = false;
    private static boolean isPushing = false;
    private static boolean isTraining = false;

    // sockets related
    String DEFAULT_IP = "192.168.43.31";//"192.168.1.103
    int DEFAULT_PORT = 8080;
    int DEFAULT_TIMEOUT = 5000;
    Communication c;

    private class AsyncTaskRunner extends AsyncTask<Void, Integer, Integer> {
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
//            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            progressBar.setVisibility(View.INVISIBLE);
        }

        @Override
        protected Integer doInBackground(Void... params) {
            return 0;
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }

        //This block executes in UI when background thread finishes
        //This is where we update the UI with our classification results
        @Override
        protected void onPostExecute(Integer result) {
            super.onPostExecute(result);
            //Hide the progress bar now that we are finished
//            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            progressBar.setVisibility(View.INVISIBLE);
        }

    }

    private class AsyncTaskPull extends AsyncTaskRunner {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            String content = ClientModel.id + " is loading model...";
            textView.setText(content);
        }

        @Override
        protected Integer doInBackground(Void... params) {
            try {
                c=Communication.connect(DEFAULT_IP,DEFAULT_PORT,DEFAULT_TIMEOUT);
                if (c == null) {
                    System.out.println("c==null");
                    return 0;
                }
                c.download("server_model.zip");
                //load model
                ClientModel.model = ModelSerializer.restoreMultiLayerNetwork(ClientModel.locateToLoadModel, false);
            } catch (IOException e) {
                e.printStackTrace();
            }
            return 0;
        }

        @Override
        protected void onPostExecute(Integer result) {
            super.onPostExecute(result);
            textView.setText(getString(R.string.pull_finish));
            isPulling = false;
        }
    }

    private class AsyncTaskPush extends AsyncTaskRunner {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            String content = ClientModel.id + " is pushing...";
            textView.setText(content);
        }

        @SuppressLint("DefaultLocale")
        @Override
        protected Integer doInBackground(Void... params) {
            try {
                System.out.println("uploading ParamTable...");
                Map<String, INDArray> map = new HashMap<>();
                Map<String, INDArray> paramTable = ClientModel.transferred_model.paramTable();
                int layer = 2;
                map.put("weight", paramTable.get(String.format("%d_W", layer)));
                map.put("bias", paramTable.get(String.format("%d_b", layer)));
                c.uploadParamTable(map);
            } catch (IOException e) {
                e.printStackTrace();
            }
            c.quit();
            return 0;
        }

        //This block executes in UI when background thread finishes
        //This is where we update the UI with our classification results
        @Override
        protected void onPostExecute(Integer result) {
            super.onPostExecute(result);
            //Update the UI with output
            textView.setText(R.string.push_finish);
            isPushing = false;
        }
    }

    private class AsyncTaskTrain extends AsyncTaskRunner {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            String content = ClientModel.id + " is training...";
            textView.setText(content);
        }

        @Override
        protected Integer doInBackground(Void... params) {
            // run training process here
            if (ClientModel.locateToSaveDataSet.length() == 0) {
                // nothing to train
                System.out.println("locateToSaveDataSet:"+ClientModel.locateToSaveDataSet);
                System.out.println("nothing to train");
                return 0;
            }
            try {
                System.out.println("AsyncTaskTrain!!!");
                ClientModel.TrainingModel(ClientModel.locateToSaveDataSet);
            } catch (IOException e) {
                e.printStackTrace();
            }

            return 0;
        }

        //This block executes in UI when background thread finishes
        //This is where we update the UI with our classification results
        @Override
        protected void onPostExecute(Integer result) {
            super.onPostExecute(result);
            //Update the UI with output
            textView.setText(R.string.train_finish);
            isTraining = false;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textView = (TextView) findViewById(R.id.textView);
        progressBar=(ProgressBar)findViewById(R.id.progressBar);
        progressBar.setVisibility(View.INVISIBLE);

        verifyStoragePermission(MainActivity.this);
//        verifyNetworkPermission(MainActivity.this);

        ClientModel.id = Settings.Secure.getString(getContentResolver(), Settings.Secure.ANDROID_ID);
        System.out.println("ANDROID_ID: "+ClientModel.id);


        // pull
        Button button_pull = (Button) findViewById(R.id.button_pull);
        button_pull.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(isPulling||isPushing||isTraining){
                    return;
                }

                //battery
                BatteryManager bm = (BatteryManager)getSystemService(BATTERY_SERVICE);
                int batLevel = bm != null ? bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) : 0;
                System.out.println("battery info:"+batLevel);
                if(batLevel<20){
                    System.out.println("Low battery. Bye...");
                    c.quit();
                }

                //network
                boolean NetworkAvailable=isNetworkAvailable();
                System.out.println("isNetworkAvailable: "+isNetworkAvailable());
                if(!NetworkAvailable) {
                    System.out.println("Network is not available. Bye...");
                    c.quit();
                }

                isPulling=true;
                AsyncTaskRunner runner=new AsyncTaskPull();
                runner.execute();
//                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
//                progressBar.setVisibility(View.VISIBLE);
            }
        });

        // push
        Button button_push = (Button) findViewById(R.id.button_push);
        button_push.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(isPulling||isPushing||isTraining){
                    return;
                }
                if (ClientModel.model == null) {
                    // model is not loaded yet
                    textView.setText(R.string.warning_pull);
                    return;
                }
                isPushing=true;
                AsyncTaskRunner runner = new AsyncTaskPush();
                runner.execute();
//                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                progressBar.setVisibility(View.VISIBLE);
            }
        });

        // train
        //try removing AsyncTask, but didn't help
        Button button_train =(Button) findViewById(R.id.button_train);
        button_train.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(isPulling||isPushing||isTraining){
                    return;
                }
                if (ClientModel.model == null) {
                    // model is not loaded yet
                    textView.setText(R.string.warning_pull);
                    return;
                }
                isTraining = true;
                String content = ClientModel.id + " is training...";
                textView.setText(content);
                if (ClientModel.locateToSaveDataSet.length() == 0) {
                    // nothing to train
                    System.out.println("locateToSaveDataSet:"+ClientModel.locateToSaveDataSet);
                    System.out.println("nothing to train");
                    isTraining = false;
                    return;
                }
                try {
                    ClientModel.TrainingModel(ClientModel.locateToSaveDataSet);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                System.out.println("train finish");
                textView.setText("train finish");
                isTraining = false;
            }
        });

        // infer
        Button button_infer =(Button) findViewById(R.id.button_infer);
        button_infer.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(isPulling||isPushing||isTraining){
                    return;
                }
                if(ClientModel.model==null){
                    // model is not loaded yet
                    textView.setText(R.string.warning_pull);
                }

            }
        });

        // Enables Always-on
        setAmbientEnabled();
    }
}
