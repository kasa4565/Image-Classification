﻿using System;
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
        static void Main()
        {
            string outputMlNetModelFilePath = GetOutputModelFilePath();
            string predictMlNetModelFilePath = GetPredictModelFilePath();
            string fullImagesetFolderPath = GetFullImagesetFolderPath();
            string imagesFolderPathForPredictions = GetImagesForPredictionFolderPath();

            var mlContext = new MLContext(seed: 1);

            // Specify MLContext Filter to only show feedback log/traces about ImageClassification
            mlContext.Log += FilterMLContextLog;

            // 2. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
            IDataView shuffledFullImageFilePathsDataset = GetShuffledFullImageFilePathsDataset(fullImagesetFolderPath, mlContext);

            // 3. Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical)
            IDataView shuffledFullImagesDataset = GetShuffledFullImagesDataset(fullImagesetFolderPath, mlContext, shuffledFullImageFilePathsDataset);

            var trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2);
            IDataView trainDataView = trainTestData.TrainSet;
            IDataView testDataView = trainTestData.TestSet;
            var pipeline = GetTrainingPipeline(mlContext, testDataView);

            // 5.1 (OPTIONAL) Define the model's training pipeline by using explicit hyper-parameters
            //
            //var options = new ImageClassificationTrainer.Options()
            //{
            //    FeatureColumnName = "Image",
            //    LabelColumnName = "LabelAsKey",
            //    // Just by changing/selecting InceptionV3/MobilenetV2/ResnetV250  
            //    // you can try a different DNN architecture (TensorFlow pre-trained model). 
            //    Arch = ImageClassificationTrainer.Architecture.MobilenetV2,
            //    Epoch = 50,       //100
            //    BatchSize = 10,
            //    LearningRate = 0.01f,
            //    MetricsCallback = (metrics) => Console.WriteLine(metrics),
            //    ValidationSet = testDataView
            //};

            //var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
            //        .Append(mlContext.Transforms.Conversion.MapKeyToValue(
            //            outputColumnName: "PredictedLabel",
            //            inputColumnName: "PredictedLabel"));

            // 6. Train/create the ML model
            ITransformer trainedModel = GetTrainedModel(trainDataView, pipeline);

            // 7. Get the quality metrics (accuracy, etc.)
            EvaluateModel(mlContext, testDataView, trainedModel);

            // 8. Save the model to assets/outputs (You get ML.NET .zip model file and TensorFlow .pb model file)
            SaveModel(outputMlNetModelFilePath, mlContext, trainDataView, trainedModel);

            //Copy model to predict
            CopyModelToPredict(outputMlNetModelFilePath, predictMlNetModelFilePath);

            // 9. Try a single prediction simulating an end-user app
            TrySinglePrediction(imagesFolderPathForPredictions, mlContext, trainedModel);

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        private static void CopyModelToPredict(string outputMlNetModelFilePath, string predictMlNetModelFilePath)
        {
            File.Copy(outputMlNetModelFilePath, predictMlNetModelFilePath, true);
            Console.WriteLine($"Model copied to: {predictMlNetModelFilePath}");
        }

        private static void SaveModel(string outputMlNetModelFilePath, MLContext mlContext, IDataView trainDataView, ITransformer trainedModel)
        {
            mlContext.Model.Save(trainedModel, trainDataView.Schema, outputMlNetModelFilePath);
            Console.WriteLine($"Model saved to: {outputMlNetModelFilePath}");
        }

        private static ITransformer GetTrainedModel(IDataView trainDataView, Microsoft.ML.Data.EstimatorChain<KeyToValueMappingTransformer> pipeline)
        {
            Console.WriteLine("*** Training the image classification model with DNN Transfer Learning on top of the selected pre-trained model/architecture ***");

            // Measuring training time
            var watch = Stopwatch.StartNew();

            //Train
            ITransformer trainedModel = pipeline.Fit(trainDataView);

            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;

            Console.WriteLine($"Training with transfer learning took: {elapsedMs / 1000} seconds");
            return trainedModel;
        }

        private static Microsoft.ML.Data.EstimatorChain<KeyToValueMappingTransformer> GetTrainingPipeline(MLContext mlContext, IDataView testDataView)
        {

            // 5. Define the model's training pipeline using DNN default values
            //
            return mlContext.MulticlassClassification.Trainers
                    .ImageClassification(featureColumnName: "Image",
                                         labelColumnName: "LabelAsKey",
                                         validationSet: testDataView)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel",
                                                                      inputColumnName: "PredictedLabel"));
        }

        private static IDataView GetShuffledFullImagesDataset(string fullImagesetFolderPath, MLContext mlContext, IDataView shuffledFullImageFilePathsDataset)
        {
            return mlContext.Transforms.Conversion.
                                MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                            .Append(mlContext.Transforms.LoadRawImageBytes(
                                                            outputColumnName: "Image",
                                                            imageFolder: fullImagesetFolderPath,
                                                            inputColumnName: "ImagePath"))
                            .Fit(shuffledFullImageFilePathsDataset)
                            .Transform(shuffledFullImageFilePathsDataset);
        }

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

        private static void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making predictions in bulk for evaluating model's quality...");

            // Measuring time
            var watch = Stopwatch.StartNew();

            var predictionsDataView = trainedModel.Transform(testDataset);

            var metrics = mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName:"LabelAsKey", predictedLabelColumnName: "PredictedLabel");
            ConsoleHelper.PrintMultiClassClassificationMetrics("TensorFlow DNN Transfer Learning", metrics);

            watch.Stop();
            var elapsed2Ms = watch.ElapsedMilliseconds;

            Console.WriteLine($"Predicting and Evaluation took: {elapsed2Ms / 1000} seconds");
        }

        private static void TrySinglePrediction(string imagesFolderPathForPredictions, MLContext mlContext, ITransformer trainedModel)
        {
            // Create prediction function to try one prediction
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

        // public static string DownloadImageSet(string imagesDownloadFolder)
        // {
        //     // get a set of images to teach the network about the new classes
        //
        //     //SINGLE SMALL FLOWERS IMAGESET (200 files)
        //     const string fileName = "flower_photos_small_set.zip";
        //     var url = $"https://mlnetfilestorage.file.core.windows.net/imagesets/flower_images/flower_photos_small_set.zip?st=2019-08-07T21%3A27%3A44Z&se=2030-08-08T21%3A27%3A00Z&sp=rl&sv=2018-03-28&sr=f&sig=SZ0UBX47pXD0F1rmrOM%2BfcwbPVob8hlgFtIlN89micM%3D";
        //     Web.Download(url, imagesDownloadFolder, fileName);
        //     Compress.UnZip(Path.Join(imagesDownloadFolder, fileName), imagesDownloadFolder);
        //
        //     //SINGLE FULL FLOWERS IMAGESET (3,600 files)
        //     //string fileName = "flower_photos.tgz";
        //     //string url = $"http://download.tensorflow.org/example_images/{fileName}";
        //     //Web.Download(url, imagesDownloadFolder, fileName);
        //     //Compress.ExtractTGZ(Path.Join(imagesDownloadFolder, fileName), imagesDownloadFolder);
        //
        //     return Path.GetFileNameWithoutExtension(fileName);
        // }

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

