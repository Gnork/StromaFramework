using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex32;
using MathNet.Numerics.Distributions;

namespace StromaDetectionRBM
{
    class WeightsHelper
    {
        public static Matrix<float> loadWeights(String filePath)
        {
            throw new NotImplementedException();
        }

        public static void saveWeights(Matrix<float> weights, String filePath)
        {
            throw new NotImplementedException();
        }

        public static Matrix<float> generateWeights(int rows, int columns, Random random)
        {
            return Matrix<float>.Build.Random(rows, columns, new Normal(0.5, 0.1, random));
        }
    }
}
