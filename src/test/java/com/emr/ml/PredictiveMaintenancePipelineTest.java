package com.emr.ml;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.apache.spark.sql.functions.col;
import static org.junit.jupiter.api.Assertions.*;

class PredictiveMaintenancePipelineTest {

    private static SparkSession spark;
    private static Dataset<Row> data;

    @BeforeAll
    static void setUp() {
        spark = SparkSession.builder()
            .master("local[2]")
            .appName("predictive-maintenance-tests")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.ui.enabled", "false")
            .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");
        data = spark.read()
            .schema(PredictiveMaintenancePipeline.SCHEMA)
            .json("sample-data/sensor_readings.json")
            .cache();
    }

    @AfterAll
    static void tearDown() {
        if (spark != null) spark.stop();
    }

    @Test
    void dataLoadsWithBothClasses() {
        assertTrue(data.count() > 0);
        assertTrue(data.filter(col("failure_within_24h").equalTo(1.0)).count() > 0);
        assertTrue(data.filter(col("failure_within_24h").equalTo(0.0)).count() > 0);
    }

    @Test
    void modelAchievesAboveRandomAUC() {
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.8, 0.2}, 42L);
        PipelineModel model = PredictiveMaintenancePipeline.buildPipeline().fit(splits[0]);
        double auc = new BinaryClassificationEvaluator()
            .setLabelCol("failure_within_24h")
            .setRawPredictionCol("rawPrediction")
            .setMetricName("areaUnderROC")
            .evaluate(model.transform(splits[1]));
        assertTrue(auc > 0.5, "AUC should beat random baseline, got: " + auc);
    }

    @Test
    void predictionsAreBinary() {
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.8, 0.2}, 42L);
        PipelineModel model = PredictiveMaintenancePipeline.buildPipeline().fit(splits[0]);
        List<Row> rows = model.transform(splits[1]).select("prediction").collectAsList();
        assertFalse(rows.isEmpty());
        for (Row r : rows) {
            double p = r.getDouble(0);
            assertTrue(p == 0.0 || p == 1.0, "Prediction must be 0 or 1, got: " + p);
        }
    }

    @Test
    void noNullFeatures() {
        long nulls = data.filter(
            col("temperature").isNull()
                .or(col("vibration").isNull())
                .or(col("rpm").isNull())
                .or(col("pressure").isNull())
        ).count();
        assertEquals(0L, nulls);
    }
}
