using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Common;
using ImageClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

namespace ImageClassification.Train
{
    internal class Program
    {
        private static MLContext mlContext;

        /// <summary>
        /// Start the program.
        /// </summary>
        static void Main()
        {
            string outputMlNetModelFilePath = GetOutputModelFilePath();
            string predictMlNetModelFilePath = GetPredictModelFilePath();
            string fullImagesetFolderPath = GetFullImagesetFolderPath();
            string imagesFolderPathForPredictions = GetImagesForPredictionFolderPath();

            mlContext = new MLContext(seed: 1);
            SpecifyContextFilter();

            IDataView shuffledFullImageFilePathsDataset = GetShuffledFullImageFilePathsDataset(fullImagesetFolderPath, mlContext);
            IDataView shuffledFullImagesDataset = GetShuffledFullImagesDataset(fullImagesetFolderPath, mlContext, shuffledFullImageFilePathsDataset);

            var trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2);
            IDataView trainDataView = trainTestData.TrainSet;
            IDataView testDataView = trainTestData.TestSet;

            var pipeline = GetTrainingPipeline(mlContext, testDataView);

            ITransformer trainedModel = GetTrainedModel(trainDataView, pipeline);
            EvaluateModel(mlContext, testDataView, trainedModel);

            SaveModel(outputMlNetModelFilePath, mlContext, trainDataView, trainedModel);
            CopyModelToPredict(outputMlNetModelFilePath, predictMlNetModelFilePath);

            TrySinglePrediction(imagesFolderPathForPredictions, mlContext, trainedModel);

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        /// <summary>
        /// Specify MLContext Filter to only show feedback log/traces about ImageClassification
        /// </summary>
        /// <param name="mlContext"></param>
        private static void SpecifyContextFilter()
        {
            mlContext.Log += FilterMLContextLog;
        }

        /// <summary>
        /// Copy model to predict.
        /// </summary>
        /// <param name="outputMlNetModelFilePath"></param>
        /// <param name="predictMlNetModelFilePath"></param>
        private static void CopyModelToPredict(string outputMlNetModelFilePath, string predictMlNetModelFilePath)
        {
            File.Copy(outputMlNetModelFilePath, predictMlNetModelFilePath, true);
            Console.WriteLine($"Model copied to: {predictMlNetModelFilePath}");
        }

        /// <summary>
        /// Save the model to assets/outputs (You get ML.NET .zip model file and TensorFlow .pb model file).
        /// </summary>
        /// <param name="outputMlNetModelFilePath"></param>
        /// <param name="mlContext"></param>
        /// <param name="trainDataView"></param>
        /// <param name="trainedModel"></param>
        private static void SaveModel(string outputMlNetModelFilePath, MLContext mlContext, IDataView trainDataView, ITransformer trainedModel)
        {
            mlContext.Model.Save(trainedModel, trainDataView.Schema, outputMlNetModelFilePath);
            Console.WriteLine($"Model saved to: {outputMlNetModelFilePath}");
        }

        /// <summary>
        /// Train/create the ML model.
        /// </summary>
        /// <param name="trainDataView"></param>
        /// <param name="pipeline"></param>
        /// <returns>Trained Model</returns>
        private static ITransformer GetTrainedModel(IDataView trainDataView, Microsoft.ML.Data.EstimatorChain<KeyToValueMappingTransformer> pipeline)
        {
            Console.WriteLine("*** Training the image classification model with DNN Transfer Learning on top of the selected pre-trained model/architecture ***");
            var watch = Stopwatch.StartNew();
            ITransformer trainedModel = pipeline.Fit(trainDataView);

            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;

            Console.WriteLine($"Training with transfer learning took: {elapsedMs / 1000} seconds");
            return trainedModel;
        }

        /// <summary>
        /// Define the model's training pipeline using DNN default values.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="testDataView"></param>
        /// <returns>Training Pipeline</returns>
        private static Microsoft.ML.Data.EstimatorChain<KeyToValueMappingTransformer> GetTrainingPipeline(MLContext mlContext, IDataView testDataView) =>
            mlContext.MulticlassClassification.Trainers
                    .ImageClassification(featureColumnName: "Image",
                                         labelColumnName: "LabelAsKey",
                                         validationSet: testDataView)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel",
                                                                      inputColumnName: "PredictedLabel"));

        /// <summary>
        /// Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical).
        /// </summary>
        /// <param name="fullImagesetFolderPath"></param>
        /// <param name="mlContext"></param>
        /// <param name="shuffledFullImageFilePathsDataset"></param>
        /// <returns>Shuffled Full Images Dataset</returns>
        private static IDataView GetShuffledFullImagesDataset(string fullImagesetFolderPath, MLContext mlContext, IDataView shuffledFullImageFilePathsDataset) =>
            mlContext.Transforms.Conversion.
                                MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                            .Append(mlContext.Transforms.LoadRawImageBytes(
                                                            outputColumnName: "Image",
                                                            imageFolder: fullImagesetFolderPath,
                                                            inputColumnName: "ImagePath"))
                            .Fit(shuffledFullImageFilePathsDataset)
                            .Transform(shuffledFullImageFilePathsDataset);

        /// <summary>
        /// Load the initial full image-set into an IDataView and shuffle so it'll be better balanced.
        /// </summary>
        /// <param name="fullImagesetFolderPath"></param>
        /// <param name="mlContext"></param>
        /// <returns>Shuffled Full Image File Paths Dataset</returns>
        private static IDataView GetShuffledFullImageFilePathsDataset(string fullImagesetFolderPath, MLContext mlContext)
        {
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: fullImagesetFolderPath, useFolderNameAsLabel: true);
            IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledFullImageFilePathsDataset = mlContext.Data.ShuffleRows(fullImagesDataset);
            return shuffledFullImageFilePathsDataset;
        }

        private static string GetImagesForPredictionFolderPath()
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            return Path.Combine(assetsPath, "inputs", "test-images");
        }

        private static string GetPredictModelFilePath()
        {
            string solutionRelativePath = @"../../../../";
            string solutionPath = GetAbsolutePath(solutionRelativePath);
            return Path.Combine(solutionPath, "ImageClassification.Predict", "assets", "inputs", "MLNETModel", "imageClassifier.zip");
        }

        private static string GetOutputModelFilePath()
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            return Path.Combine(assetsPath, "outputs", "imageClassifier.zip");
        }

        private static string GetFullImagesetFolderPath()
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs", "images");
            string finalImagesFolderName = "photos";
            string fullImagesetFolderPath = Path.Combine(imagesDownloadFolderPath, finalImagesFolderName);

            return fullImagesetFolderPath;
        }

        /// <summary>
        /// Get the quality metrics (accuracy, etc.).
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="testDataset"></param>
        /// <param name="trainedModel"></param>
        private static void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making predictions in bulk for evaluating model's quality...");

            var watch = Stopwatch.StartNew();

            var predictionsDataView = trainedModel.Transform(testDataset);

            var metrics = mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName:"LabelAsKey", predictedLabelColumnName: "PredictedLabel");
            ConsoleHelper.PrintMultiClassClassificationMetrics("TensorFlow DNN Transfer Learning", metrics);

            watch.Stop();
            var elapsed2Ms = watch.ElapsedMilliseconds;

            Console.WriteLine($"Predicting and Evaluation took: {elapsed2Ms / 1000} seconds");
        }

        /// <summary>
        /// Try a single prediction simulating an end-user app.
        /// </summary>
        /// <param name="imagesFolderPathForPredictions"></param>
        /// <param name="mlContext"></param>
        /// <param name="trainedModel"></param>
        private static void TrySinglePrediction(string imagesFolderPathForPredictions, MLContext mlContext, ITransformer trainedModel)
        {
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<InMemoryImageData, ImagePrediction>(trainedModel);

            var testImages = FileUtils.LoadInMemoryImagesFromDirectory(
                imagesFolderPathForPredictions, false);

            var imageToPredict = testImages.First();

            var prediction = predictionEngine.Predict(imageToPredict);

            Console.WriteLine(
                $"Image Filename : [{imageToPredict.ImageFileName}], " +
                $"Scores : [{string.Join(",", prediction.Score)}], " +
                $"Predicted Label : {prediction.PredictedLabel}");
        }


        public static IEnumerable<ImageData> LoadImagesFromDirectory(
            string folder,
            bool useFolderNameAsLabel = true)
            => FileUtils.LoadImagesFromDirectory(folder, useFolderNameAsLabel)
                .Select(x => new ImageData(x.imagePath, x.label));

        public static string GetAbsolutePath(string relativePath)
            => FileUtils.GetAbsolutePath(typeof(Program).Assembly, relativePath);

        public static void ConsoleWriteImagePrediction(string ImagePath, string Label, string PredictedLabel, float Probability)
        {
            var defaultForeground = Console.ForegroundColor;
            var labelColor = ConsoleColor.Magenta;
            var probColor = ConsoleColor.Blue;

            Console.Write("Image File: ");
            Console.ForegroundColor = labelColor;
            Console.Write($"{Path.GetFileName(ImagePath)}");
            Console.ForegroundColor = defaultForeground;
            Console.Write(" original labeled as ");
            Console.ForegroundColor = labelColor;
            Console.Write(Label);
            Console.ForegroundColor = defaultForeground;
            Console.Write(" predicted as ");
            Console.ForegroundColor = labelColor;
            Console.Write(PredictedLabel);
            Console.ForegroundColor = defaultForeground;
            Console.Write(" with score ");
            Console.ForegroundColor = probColor;
            Console.Write(Probability);
            Console.ForegroundColor = defaultForeground;
            Console.WriteLine("");
        }

        private static void FilterMLContextLog(object sender, LoggingEventArgs e)
        {
            if (e.Message.StartsWith("[Source=ImageClassificationTrainer;"))
            {
                Console.WriteLine(e.Message);
            }
        }
    }
}

