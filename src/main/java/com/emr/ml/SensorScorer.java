package com.emr.ml;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class SensorScorer {

    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("Usage: SensorScorer <model_path> <input_path> <output_path>");
            System.exit(1);
        }
        String modelPath  = args[0];
        String inputPath  = args[1];
        String outputPath = args[2];

        SparkSession spark = SparkSession.builder()
            .appName("SensorScorer")
            .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
            .getOrCreate();

        try {
            PipelineModel model = PipelineModel.load(modelPath);

            Dataset<Row> newData = spark.read()
                .schema(PredictiveMaintenancePipeline.SCHEMA)
                .json(inputPath);

            model.transform(newData)
                .select("sensor_id", "machine_id", "timestamp",
                        "prediction", "probability")
                .write()
                .mode("overwrite")
                .parquet(outputPath);

        } finally {
            spark.stop();
        }
    }
}
