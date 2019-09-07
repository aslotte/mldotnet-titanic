using Microsoft.ML.Data;

namespace TitanicPredictions.Train.Schema
{
    public sealed class SurvivalPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Survived { get; set; }
    }
}
