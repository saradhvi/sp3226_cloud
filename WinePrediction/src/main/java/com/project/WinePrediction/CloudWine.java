package com.project.WinePrediction;

import org.apache.spark.sql.functions;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import java.io.IOException;


public class CloudWine {
    public static void main(String[] args) {
        String training = "";
        String test = "";
        String out = "";
        if (args.length > 3) {
            System.err.println("Error occured");
            System.exit(1);
        } else if(args.length ==3){
            training = args[0];
            test = args[1];
            out = args[2] + "model";
        } else{
            training = "s3://tb369/TrainingDataset.csv";
            test = "s3://sp3226/ValidationDataset.csv";
            out = "s3://sp3226/Test.model";
        
        }
        
          
        SparkSession spark = SparkSession.builder()
        .appName("WineQualityPrediction")
        .config("spark.master", "local")
        .getOrCreate();
        
        // Create a JavaSparkContext object 
        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        spark.sparkContext().setLogLevel("ERROR");

        
        //  Training data file from AWS S3, converting it to a DataFrame.
        Dataset<Row> trainingData = spark.read().format("csv")
        .option("header", true)
        .option("quote", "\"") // handle escaped quotes
        .option("delimiter", ";")
        .load(training);
        
        
        
        
        Dataset<Row> testData = spark.read().format("csv")
        .option("header", true)
        .option("quote", "\"") // handle escaped quotes
        .option("delimiter", ";")
        .load(test);

        String[] inputColumns = {"fixed acidity", "volatile acidity", "citric acid", "chlorides", "total sulfur dioxide", "density", "sulphates", "alcohol"};
        
      
        for (String col : inputColumns) {
            trainingData = trainingData.withColumn(col, trainingData.col(col).cast("Double")); 
        }
        
      
        for (String col : inputColumns) {
            testData = testData.withColumn(col, testData.col(col).cast("Double"));
        }
        
   
        trainingData = trainingData.withColumn("quality", trainingData.col("quality").cast("Double"));
        testData = testData.withColumn("quality", testData.col("quality").cast("Double"));
        
        trainingData = trainingData.withColumn("label", functions.when(trainingData.col("quality").geq(7), 1.0).otherwise(0.0));
        testData = testData.withColumn("label", functions.when(testData.col("quality").geq(7), 1.0).otherwise(0.0));
        
        
        VectorAssembler assembler = new VectorAssembler()
        .setInputCols(inputColumns)
        .setOutputCol("features");
        
        // Create a RandomForestClassifier it is best suited for multiclass classification
        RandomForestClassifier rf = new RandomForestClassifier()
        .setLabelCol("quality")
        .setFeaturesCol("features")
        .setNumTrees(320)
        .setMaxBins(14)
        .setMaxDepth(26)
        .setSeed(150)
        .setImpurity("entropy");
        
        // Configure an ML pipeline, which consists of assembler and random forest classifier.
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, rf});
        
        
        PipelineModel model = pipeline.fit(trainingData);
        
        System.out.println(model);
        
       
        Dataset<Row> predictions = model.transform(testData);
        
        Selecting rows to display.
        predictions.select("prediction", "quality", "features").show(2);
        
        
        MulticlassClassificationEvaluator mcevaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("quality")
        .setPredictionCol("prediction");
        
        System.out.println(mcevaluator);
        
        double logloss = mcevaluator.setMetricName(" logLoss").evaluate(predictions);
        
        double accuracy = mcevaluator.setMetricName("accuracy").evaluate(predictions);
        double truepositiverate = mcevaluator.setMetricName(" weightedTruePositiveRate").evaluate(predictions);
        
        
        
        System.out.println("Logloss is " + logloss);
        System.out.println("Accuracy  is " + accuracy);
        System.out.println("Truepositive rate  is " + truepositiverate);

        try {
            model.write().overwrite().save(out);
        } catch (IOException e) {
            System.err.println("Failed to save the model: " + e.getMessage());
        }
    }
}
