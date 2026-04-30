package com.emr.ml;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

public class PredictiveMaintenancePipeline {

    static final String[] FEATURE_COLS = {
        "temperature", "vibration", "rpm", "pressure", "age_days", "hours_since_maintenance"
    };

    public static final StructType SCHEMA = new StructType()
        .add("sensor_id",               DataTypes.StringType,    false)
        .add("machine_id",              DataTypes.StringType,    false)
        .add("timestamp",               DataTypes.TimestampType, false)
        .add("temperature",             DataTypes.DoubleType,    false)
        .add("vibration",               DataTypes.DoubleType,    false)
        .add("rpm",                     DataTypes.DoubleType,    false)
        .add("pressure",                DataTypes.DoubleType,    false)
        .add("age_days",                DataTypes.IntegerType,   false)
        .add("hours_since_maintenance", DataTypes.IntegerType,   false)
        .add("failure_within_24h",      DataTypes.DoubleType,    false);

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Usage: PredictiveMaintenancePipeline <input_path> <output_path>");
            System.exit(1);
        }
        String inputPath  = args[0];
        String outputPath = args[1];

        SparkSession spark = SparkSession.builder()
            .appName("PredictiveMaintenancePipeline")
            .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
            .getOrCreate();

        try {
            Dataset<Row> data = spark.read()
                .schema(SCHEMA)
                .json(inputPath)
                .cache();

            Dataset<Row>[] splits = data.randomSplit(new double[]{0.8, 0.2}, 42L);
            PipelineModel model = buildPipeline().fit(splits[0]);

            Dataset<Row> predictions = model.transform(splits[1]);

            double auc = new BinaryClassificationEvaluator()
                .setLabelCol("failure_within_24h")
                .setRawPredictionCol("rawPrediction")
                .setMetricName("areaUnderROC")
                .evaluate(predictions);

            System.out.printf("Test AUC: %.4f%n", auc);

            predictions
                .select("sensor_id", "machine_id", "timestamp",
                        "failure_within_24h", "prediction", "probability")
                .write().mode("overwrite")
                .parquet(outputPath + "/predictions/");

            spark.sql(String.format("SELECT %.6f AS auc", auc))
                .write().mode("overwrite")
                .json(outputPath + "/metrics/");

            try {
                model.write().overwrite().save(outputPath + "/model/");
            } catch (java.io.IOException e) {
                throw new RuntimeException("Failed to save model", e);
            }

        } finally {
            spark.stop();
        }
    }

    static Pipeline buildPipeline() {
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(FEATURE_COLS)
            .setOutputCol("raw_features");

        StandardScaler scaler = new StandardScaler()
            .setInputCol("raw_features")
            .setOutputCol("features")
            .setWithMean(true)
            .setWithStd(true);

        LogisticRegression lr = new LogisticRegression()
            .setLabelCol("failure_within_24h")
            .setFeaturesCol("features")
            .setMaxIter(100)
            .setRegParam(0.01);

        return new Pipeline().setStages(new PipelineStage[]{assembler, scaler, lr});
    }
}
