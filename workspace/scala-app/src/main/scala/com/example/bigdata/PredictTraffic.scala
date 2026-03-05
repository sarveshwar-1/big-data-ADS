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
    // Model now expects prev_count, hour_sin, hour_cos
    val pi = math.Pi
    val liveData = Seq(
      (2000, math.sin(2*pi*10/24), math.cos(2*pi*10/24)),
      (5000, math.sin(2*pi*14/24), math.cos(2*pi*14/24)),
      (100,  math.sin(2*pi*3/24),  math.cos(2*pi*3/24)),
      (8000, math.sin(2*pi*18/24), math.cos(2*pi*18/24))
    ).toDF("prev_count", "hour_sin", "hour_cos")

    println("Predicting future traffic...")
    val predictions = model.transform(liveData)

    println("Traffic Forecast:")
    predictions.select("prev_count", "prediction").show()

    println("Inference Job Complete.")
    spark.stop()
  }
}
