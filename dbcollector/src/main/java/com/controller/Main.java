package com.controller;

import com.controller.collectors.DBCollector;
import com.controller.collectors.PostgresCollector;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

/**
 * Controller main.
 * @author Shuli
 */
public class Main {
    private static final int DEFAULT_TIME = 5000;  //default observation time: 5000 ms
    public static void main(String[] args) {
        // Parse command line argument
        if(args.length % 2 != 0) {
            System.out.println("Command line argument mal formed");
            return;
        }
        int time = DEFAULT_TIME; // set time to default
        for(int i = 0; i < args.length; i += 2){
            String flag = args[i];
            String val = args[++i];
            switch (flag) {
                case "-t" :
                    time = Integer.valueOf(val);
                    break;
                default:
                    System.out.println("Invalid flag: " + flag);
                    break;
            }
        }

        // Parse input config file
        String configFileName = "input_config.json";
        HashMap<String, String> input = ConfigFileParser.getInputFromConfigFile(configFileName);
        // db type
        String dbtype = input.get("database_type");
        DBCollector collector = null;
        // parameters for creating a collector.
        String username = input.get("username");
        String password = input.get("password");
        String dbURL = input.get("database_url");
        // uploader
        String uploadCode = input.get("upload_code");
        String uploadURL = input.get("upload_url");

        switch (dbtype) {
            case "PostgreSQL":
                collector = new PostgresCollector(dbURL, username, password);
                break;
            default:
                System.out.println("Please specify database type!");
                return;
        }

        try {
            // summary file
            PrintWriter summaryWriter = new PrintWriter("out.summary", "UTF-8");
            summaryWriter.println("Observation time: " + time + " ms");
            summaryWriter.println("Database type: " + dbtype);
            summaryWriter.println("Database version: " + collector.collectVersion());

            // first collection (before queries)
            PrintWriter metricsWriter = new PrintWriter("metrics_before.json", "UTF-8");
            metricsWriter.println(collector.collectMetrics());
            metricsWriter.flush();
            metricsWriter.close();
            PrintWriter knobsWriter = new PrintWriter("knobs_before.json", "UTF-8");
            knobsWriter.println(collector.collectParameters());
            knobsWriter.flush();
            knobsWriter.close();

            // record start time
            summaryWriter.println("\nExecution Summary: ");
            summaryWriter.println("start time in milliseconds: " + System.currentTimeMillis());
            summaryWriter.println("start time in UTC: " + Instant.now());

            // go to sleep
            Thread.sleep(time);

            // record end time
            summaryWriter.println("end time in milliseconds: " + System.currentTimeMillis());
            summaryWriter.println("end time in UTC: " + Instant.now());
            summaryWriter.flush();

            // second collection (after queries)
            PrintWriter metricsWriterFinal = new PrintWriter("metrics_after.json", "UTF-8");
            metricsWriterFinal.println(collector.collectMetrics());
            metricsWriterFinal.flush();
            metricsWriterFinal.close();
            PrintWriter knobsWriterFinal = new PrintWriter("knobs_after.json", "UTF-8");
            knobsWriterFinal.println(collector.collectParameters());
            knobsWriterFinal.flush();
            knobsWriterFinal.close();
        } catch (FileNotFoundException | UnsupportedEncodingException | InterruptedException e) {
            e.printStackTrace();
        }

        Map<String, String> outfiles = new HashMap<>();
        outfiles.put("knobs_before", "knobs_before.json");
        outfiles.put("knobs_after", "knobs_after.json");
        outfiles.put("metrics_before", "metrics_before.json");
        outfiles.put("metrics_after", "metrics_after.json");
        outfiles.put("summary", "out.summary");
        ResultUploader.upload(uploadURL, uploadCode, outfiles);

    }
}
