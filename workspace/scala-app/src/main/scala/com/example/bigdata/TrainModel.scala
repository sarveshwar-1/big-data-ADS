package com.example.bigdata

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object TrainModel {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("TrafficPredictionModel")
      .getOrCreate()

    import spark.implicits._

    val inputPath = "hdfs://namenode:8020/logs/features/traffic_ml_ready"
    println(s"Loading features from $inputPath...")
    val data = spark.read.parquet(inputPath)

    val assembler = new VectorAssembler()
      .setInputCols(Array("prev_count"))
      .setOutputCol("features")

    println("Splitting data into Training (80%) and Testing (20%)...")
    val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 42)

    println("Training Linear Regression Model...")
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("request_count")

    val pipeline = new Pipeline()
      .setStages(Array(assembler, lr))

    val model = pipeline.fit(trainData)

    println("Evaluating model on Test Data...")
    val predictions = model.transform(testData)

    val rmseEvaluator = new RegressionEvaluator()
      .setLabelCol("request_count")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = rmseEvaluator.evaluate(predictions)

    val r2Evaluator = new RegressionEvaluator()
      .setLabelCol("request_count")
      .setPredictionCol("prediction")
      .setMetricName("r2")
    val r2 = r2Evaluator.evaluate(predictions)

    val maeEvaluator = new RegressionEvaluator()
      .setLabelCol("request_count")
      .setPredictionCol("prediction")
      .setMetricName("mae")
    val mae = maeEvaluator.evaluate(predictions)

    println(s"Root Mean Squared Error (RMSE): $rmse")
    println(s"Mean Absolute Error     (MAE):  $mae")
    println(s"R² Score:                        $r2")

    println("Sample Predictions:")
    predictions.select("window_start", "prev_count", "request_count", "prediction").show(5)

    val modelPath = "hdfs://namenode:8020/models/traffic_prediction_v1"
    println(s"Saving trained model to $modelPath...")
    model.write.overwrite().save(modelPath)

    val metricsPath = "hdfs://namenode:8020/models/metrics/lr_metrics"
    println(s"Saving LR metrics to $metricsPath...")
    val metricsDF = Seq((rmse, mae, r2)).toDF("rmse", "mae", "r2")
    metricsDF.coalesce(1).write.mode("overwrite").json(metricsPath)

    println("Model Training Complete.")
    spark.stop()
  }
}
