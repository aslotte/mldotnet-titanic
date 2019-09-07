using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using TitanicPredictions.Train.Schema;

namespace TitanicPredictions.Train
{
    class Program
    {
        private const string DataPath = "Data/data.csv";

        public static void Main(string[] args)
        {
            Console.WriteLine("Starting to train model");
            var mlContext = new MLContext(seed: 1);

            //Load
            Console.WriteLine("Reading data...");
            var data = mlContext.Data.LoadFromTextFile<Passenger>(DataPath, hasHeader: true, separatorChar: ',');
            var testTrainDataSet = mlContext.Data.TrainTestSplit(data);

            //Transform
            Console.WriteLine("Transforming data...");
            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding("Sex")
                .Append(mlContext.Transforms.Text.FeaturizeText("Name"))
                .Append(mlContext.Transforms.ReplaceMissingValues("Age", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
                .Append(mlContext.Transforms.Concatenate("Features", "Pclass", "Sex", "Name",
                        "SiblingsAboard", "ParentsAboard")));

            //Train
            Console.WriteLine("Training data...");
            var trainingPipeline = dataProcessPipeline
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Survived"));
            ITransformer trainedModel = trainingPipeline.Fit(testTrainDataSet.TrainSet);

            //Evaluate
            var predictions = trainedModel.Transform(testTrainDataSet.TestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Survived");

            //Print metrics
            PrintBinaryClassificationMetrics(trainingPipeline.ToString(), metrics);

            //Save model
            mlContext.Model.Save(trainedModel, testTrainDataSet.TrainSet.Schema, "model.zip");

            Console.ReadLine();
        }

        public static void PrintBinaryClassificationMetrics(string name, CalibratedBinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics for {name} binary classification model      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
            Console.WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");
            Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
            Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
            Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
            Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
            Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
            Console.WriteLine($"************************************************************");
        }
    }
}
