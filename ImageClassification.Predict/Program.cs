using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ImageClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageClassification.Predict
{
    internal class Program
    {
        /// <summary>
        /// Start the program.
        /// </summary>
        private static void Main()
        {
            string assetsPath = GetAssetsPath();
            string imagesFolderPathForPredictions = GetImagesPath(assetsPath);
            string imageClassifierModelZipFilePath = GetClassifierModelPath(assetsPath);

            try
            {
                var mlContext = new MLContext(seed: 1);
                Console.WriteLine($"Loading model from: {imageClassifierModelZipFilePath}");
                ITransformer loadedModel = GetLoadedModel(imageClassifierModelZipFilePath, mlContext);
                var predictionEngine = GetPredictionEngine(mlContext, loadedModel);
                var imagesToPredict = GetImagesToPredict(imagesFolderPathForPredictions);
                var imageToPredict = imagesToPredict.First();
                var prediction = GetFirstPrediction(predictionEngine, imageToPredict);
                DoSecondPrediction(predictionEngine, imageToPredict);
                DoubleCheckUsingIndex(predictionEngine, prediction);
                PrintInformationAboutPrediction(imageToPredict, prediction);
                PredictAllImages(predictionEngine, imagesToPredict);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.WriteLine("Press any key to end the app..");
            Console.ReadKey();
        }

        private static string GetClassifierModelPath(string assetsPath) => Path.Combine(assetsPath, "inputs", "MLNETModel", "imageClassifier.zip");

        private static string GetImagesPath(string assetsPath) => Path.Combine(assetsPath, "inputs", "images-for-predictions");

        private static string GetAssetsPath()
        {
            const string assetsRelativePath = @"../../../assets";
            var assetsPath = GetAbsolutePath(assetsRelativePath);
            return assetsPath;
        }

        /// <summary>
        /// Predict all images in the folder.
        /// </summary>
        /// <param name="predictionEngine"></param>
        /// <param name="imagesToPredict"></param>
        private static void PredictAllImages(PredictionEngine<InMemoryImageData, ImagePrediction> predictionEngine, IEnumerable<InMemoryImageData> imagesToPredict)
        {
            //Predict all images in the folder
            //
            Console.WriteLine("");
            Console.WriteLine("Predicting several images...");

            foreach (var currentImageToPredict in imagesToPredict)
            {
                var currentPrediction = predictionEngine.Predict(currentImageToPredict);
                PrintInformationAboutPrediction(currentImageToPredict, currentPrediction);
            }
        }

        private static void PrintInformationAboutPrediction(InMemoryImageData imageToPredict, ImagePrediction prediction)
        {
            Console.WriteLine($"Image Filename : [{imageToPredict.ImageFileName}], " +
                                              $"Predicted Label : [{prediction.PredictedLabel}], " +
                                              $"Probability : [{prediction.Score.Max()}] "
                                              );
        }

        /// <summary>
        /// Double-check using the index.
        /// </summary>
        /// <param name="predictionEngine"></param>
        /// <param name="prediction"></param>
        private static void DoubleCheckUsingIndex(PredictionEngine<InMemoryImageData, ImagePrediction> predictionEngine, ImagePrediction prediction)
        {
            ////////
            // Double-check using the index
            var maxIndex = prediction.Score.ToList().IndexOf(prediction.Score.Max());
            VBuffer<ReadOnlyMemory<char>> keys = default;
            predictionEngine.OutputSchema[3].GetKeyValues(ref keys);
            var keysArray = keys.DenseValues().ToArray();
            var predictedLabelString = keysArray[maxIndex];
            ////////
        }

        /// <summary>
        /// Do second prediction for count the taken time.
        /// </summary>
        /// <param name="predictionEngine"></param>
        /// <param name="imageToPredict"></param>
        private static void DoSecondPrediction(PredictionEngine<InMemoryImageData, ImagePrediction> predictionEngine, InMemoryImageData imageToPredict)
        {
            var watch2 = System.Diagnostics.Stopwatch.StartNew();

            predictionEngine.Predict(imageToPredict);

            watch2.Stop();
            var elapsedMs2 = watch2.ElapsedMilliseconds;
            Console.WriteLine("Second Prediction took: " + elapsedMs2 + "mlSecs");
        }

        /// <summary>
        /// Get the prediction and count the taken time.
        /// </summary>
        /// <param name="predictionEngine"></param>
        /// <param name="imageToPredict"></param>
        /// <returns>Prediction</returns>
        private static ImagePrediction GetFirstPrediction(PredictionEngine<InMemoryImageData, ImagePrediction> predictionEngine, InMemoryImageData imageToPredict)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();

            var prediction = predictionEngine.Predict(imageToPredict);

            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("First Prediction took: " + elapsedMs + "mlSecs");
            return prediction;
        }

        private static IEnumerable<InMemoryImageData> GetImagesToPredict(string imagesFolderPathForPredictions)
        {
            return FileUtils.LoadInMemoryImagesFromDirectory(imagesFolderPathForPredictions, false);
        }

        /// <summary>
        /// Create prediction engine to try a single prediction (input = ImageData, output = ImagePrediction).
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="loadedModel"></param>
        /// <returns>Prediction Engine</returns>
        private static PredictionEngine<InMemoryImageData, ImagePrediction> GetPredictionEngine(MLContext mlContext, ITransformer loadedModel)
        {

            // Create prediction engine to try a single prediction (input = ImageData, output = ImagePrediction)
            return mlContext.Model.CreatePredictionEngine<InMemoryImageData, ImagePrediction>(loadedModel);
        }

        /// <summary>
        /// Load the model.
        /// </summary>
        /// <param name="imageClassifierModelZipFilePath"></param>
        /// <param name="mlContext"></param>
        /// <returns>Loaded model</returns>
        private static ITransformer GetLoadedModel(string imageClassifierModelZipFilePath, MLContext mlContext)
        {

            // Load the model
            return mlContext.Model.Load(imageClassifierModelZipFilePath, out var modelInputSchema);
        }

        public static string GetAbsolutePath(string relativePath)
            => FileUtils.GetAbsolutePath(typeof(Program).Assembly, relativePath);
    }
}
