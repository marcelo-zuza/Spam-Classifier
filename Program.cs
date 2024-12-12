using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML;

namespace SpamClassifier
{
    class Prgram
    {
        public class MessageData


        {
            [LoadColumn(0)]
            public string Label { get; set; } // Span or not

            [LoadColumn(1)]
            public string Message { get; set; } // Message text
        }

        // classe pra representar aa previsão
        public class SpamPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool IsSpan { get; set; }


            public float Probability { get; set; }
            public float Score { get; set; }
        }



        static void Main(string[] args)
        {
            // criar contexto do ML.NET
            var mlContext = new MLContext();

            // caminho para o arquivo de dados
            string dataPath = "data/spamdata.csv";

            // carregar dados
            IDataView dataView = mlContext.Data.LoadFromTextFile<MessageData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ','
            );

            // preparar o pipeline de treinamento
            var pipeline = mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnName: nameof(MessageData.Message)).Append(mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "LabelKey",
                    inputColumnName: nameof(MessageData.Label)
                )).Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    labelColumnName: "LabelKey",
                    featureColumnName: "Features"
                )).Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: nameof(SpamPrediction.IsSpan), 
                    inputColumnName: "PredictedLabel"));
            
            // dividir os dados entre treino e teste
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // treinar modelo
            Console.WriteLine("Treinando modelo...");
            var model = pipeline.Fit(split.TrainSet);

            // Avaliar modelo
            Console.WriteLine("Avaliando modelo...");
            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(
                predictions,
                labelColumnName: "LabelKey",
                predictedLabelColumnName: "PredictedLabel"
            );

            Console.WriteLine($"Log-Loss: {metrics.LogLoss}");
            Console.WriteLine($"Accuracy: {metrics.MacroAccuracy}");

            // fazer previsões em novos dados
            var predictor = mlContext.Model.CreatePredictionEngine<MessageData
            , SpamPrediction>(model
            );

            var sample = new MessageData { Message = "Você ganhou um prêmio! Clique aqui para receber."};
            var result = predictor.Predict(sample);

            Console.WriteLine($"Mensagem: {sample.Message}");
            Console.WriteLine($"É Spam? {result.IsSpan} (Probabilidade: {result.Probability:P2})");

        }
    }
}