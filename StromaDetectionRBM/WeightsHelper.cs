using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace StromaDetectionRBM
{
    class WeightsHelper
    {
        public static Matrix<float> loadWeights(String filePath)
        {
            FileStream fileStream = new FileStream(@filePath, FileMode.Open);
            BinaryFormatter formatter = new BinaryFormatter();
            Matrix<float> weights = (Matrix<float>) formatter.Deserialize(fileStream);
            fileStream.Close();
            return weights;
        }

        public static void saveWeights(Matrix<float> weights, String filePath)
        {
            FileStream fileStream = new FileStream(@filePath, FileMode.Create);
            BinaryFormatter formatter = new BinaryFormatter();
            formatter.Serialize(fileStream, weights);
            fileStream.Close();
        }

        public static Matrix<float> generateWeights(int rows, int columns, Random random)
        {
            return Matrix<float>.Build.Random(rows, columns, new Normal(0.0, 0.1, random));
        }
    }
}
