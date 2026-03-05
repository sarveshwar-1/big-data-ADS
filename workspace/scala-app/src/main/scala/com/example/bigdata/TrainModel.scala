package com.example.bigdata

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

object TrainModel {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("TrafficPredictionModel")
      .getOrCreate()

    import spark.implicits._

    val inputPath = "hdfs://namenode:8020/logs/features/traffic_ml_ready_hourly"
    println(s"Loading hourly features from $inputPath...")
    val data = spark.read.parquet(inputPath)

    // Sort by timestamp and add row index for chronological split
    val sortedData = data.orderBy("hour_timestamp")
    val indexedData = sortedData.withColumn("_row_idx",
      row_number().over(Window.orderBy("hour_timestamp")) - 1)

    val totalRows = indexedData.count()
    val trainSize = (totalRows * 0.8).toLong

    println(s"Chronological split: train=$trainSize, test=${totalRows - trainSize} (total=$totalRows)")
    val trainData = indexedData.filter(col("_row_idx") < trainSize).drop("_row_idx")
    val testData = indexedData.filter(col("_row_idx") >= trainSize).drop("_row_idx")

    // Add cyclical hour features
    val pi = math.Pi
    val addHourFeatures = (df: org.apache.spark.sql.DataFrame) =>
      df.withColumn("hour_of_day", hour(col("hour_timestamp")))
        .withColumn("hour_sin", sin(lit(2.0 * pi) * col("hour_of_day") / lit(24.0)))
        .withColumn("hour_cos", cos(lit(2.0 * pi) * col("hour_of_day") / lit(24.0)))

    val trainWithFeatures = addHourFeatures(trainData)
    val testWithFeatures = addHourFeatures(testData)

    val assembler = new VectorAssembler()
      .setInputCols(Array("prev_count", "hour_sin", "hour_cos"))
      .setOutputCol("features")

    println("Training Linear Regression Model...")
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("request_count")

    val pipeline = new Pipeline()
      .setStages(Array(assembler, lr))

    val model = pipeline.fit(trainWithFeatures)

    println("Evaluating model on Test Data...")
    val predictions = model.transform(testWithFeatures)

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

    // Compute MAPE manually
    val mapeDF = predictions.filter(col("request_count") =!= 0)
      .withColumn("ape", abs(col("request_count") - col("prediction")) / col("request_count"))
    val mape = mapeDF.agg(avg("ape")).first().getDouble(0) * 100.0

    println(s"Root Mean Squared Error (RMSE): $rmse")
    println(s"Mean Absolute Error     (MAE):  $mae")
    println(s"R² Score:                        $r2")
    println(s"MAPE (%%):                       $mape")

    println("Sample Predictions:")
    predictions.select("hour_timestamp", "prev_count", "request_count", "prediction").show(5)

    val modelPath = "hdfs://namenode:8020/models/traffic_prediction_v1"
    println(s"Saving trained model to $modelPath...")
    model.write.overwrite().save(modelPath)

    val metricsPath = "hdfs://namenode:8020/models/metrics/lr_metrics"
    println(s"Saving LR metrics to $metricsPath...")
    val metricsDF = Seq((rmse, mae, r2, mape)).toDF("rmse", "mae", "r2", "mape")
    metricsDF.coalesce(1).write.mode("overwrite").json(metricsPath)

    println("Model Training Complete.")
    spark.stop()
  }
}
