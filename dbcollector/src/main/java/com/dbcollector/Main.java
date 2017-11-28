package com.dbcollector;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

/**
 * Created by MacadamiaKitten on 11/20/17.
 */
public class Main {
    public static void main(String[] args) {
        String oriDBUrl = "jdbc:postgresql://localhost:5432/postgres";
        String username = "MacadamiaKitten";
        String password = "";
        PostgresCollector postgresCollector = new PostgresCollector(oriDBUrl, username, password);

        try {
            PrintWriter metricsWriter = new PrintWriter("metrics.json", "UTF-8");
            metricsWriter.println(postgresCollector.collectMetrics());
            metricsWriter.flush();
            metricsWriter.close();
            PrintWriter knobsWriter = new PrintWriter("knobs.json", "UTF-8");
            knobsWriter.println(postgresCollector.collectParameters());
            knobsWriter.flush();
            knobsWriter.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            System.out.println(e);
        }

    }
}
