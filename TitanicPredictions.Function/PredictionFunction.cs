using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Newtonsoft.Json;
using System;
using System.IO;
using System.Threading.Tasks;
using TitanicPredictions.Train.Schema;

namespace TitanicPredictions.Function
{
    public static class PredictionFunction
    {
        [FunctionName("PredictionFunction")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "get", "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            //Get the data
            var requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            Passenger data = JsonConvert.DeserializeObject<Passenger>(requestBody);

            var mlContext = new MLContext();

            //Load the model
            ITransformer model = mlContext.Model.Load("Model/model.zip", out var _);

            //Create a prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<Passenger, SurvivalPrediction>(model);

            //Make a prediction
            SurvivalPrediction prediction = predictionEngine.Predict(data);
            var survived = Convert.ToBoolean(prediction.Survived) ? "The passenger survived" : "The passenger did not survive";

            return (ActionResult)new OkObjectResult(survived);
        }
    }
}
