package com.example.bigdata

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

object PredictTraffic {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("TrafficInference")
      .getOrCreate()

    import spark.implicits._

    val modelPath = "hdfs://namenode:8020/models/traffic_prediction_v1"
    println(s"Loading model from $modelPath...")
    val model = PipelineModel.load(modelPath)

    println("Generating 'live' data for inference...")
    val liveData = Seq(
      Tuple1(2000),
      Tuple1(5000),
      Tuple1(100),
      Tuple1(8000)
    ).toDF("prev_count")

    println("Predicting future traffic...")
    val predictions = model.transform(liveData)

    println("Traffic Forecast:")
    predictions.select("prev_count", "prediction").show()

    println("Inference Job Complete.")
    spark.stop()
  }
}
