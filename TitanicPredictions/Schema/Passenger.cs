using Microsoft.ML.Data;

namespace TitanicPredictions.Train.Schema
{
    public sealed class Passenger
    {
        [LoadColumn(0)]
        public bool Survived { get; set; }

        [LoadColumn(1)]
        public float Pclass { get; set; }

        [LoadColumn(2)]
        public string Name { get; set; }

        [LoadColumn(3)]
        public string Sex { get; set; }

        [LoadColumn(4)]
        public float Age { get; set; }

        [LoadColumn(5)]
        public float SiblingsAboard { get; set; }

        [LoadColumn(6)]
        public float ParentsAboard { get; set; }
    }
}
