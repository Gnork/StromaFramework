using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace StromaDetectionRBM
{
    class MatrixHelper
    {
        public static Matrix<float> avgMatrix(Matrix<float>[] matrices)
        {
            Matrix<float> avgMatrix = Matrix<float>.Build.Dense(matrices[0].RowCount, matrices[0].RowCount, 0.0f);

            for (int i = 0; i < matrices.Length; ++i)
            {
                avgMatrix.Add(matrices[i], avgMatrix);
            }
            avgMatrix.Multiply(1.0f / matrices.Length, avgMatrix);

            return avgMatrix;
        }

        public static Matrix<float> addLabels(Matrix<float> matrix)
        {
            int batchSize = matrix.RowCount;
            int numOfPositive = batchSize / 2;

            Matrix<float> result = Matrix<float>.Build.Dense(batchSize, 1);

            float[] labelColumn = new float[batchSize];

            int i = 0;

            for (; i < numOfPositive; ++i)
            {
                labelColumn[i] = 1.0f;
            }

            result.SetColumn(0, labelColumn);

            return matrix.Append(result);
        }

        public static Matrix<float> addEmptyLabels(Matrix<float> matrix)
        {
            int batchSize = matrix.RowCount;

            Matrix<float> result = Matrix<float>.Build.Dense(batchSize, 1, 0.5f);

            return matrix.Append(result);
        }

        public static Matrix<float> removeLabels(Matrix<float> matrix)
        {
            return matrix.RemoveColumn(matrix.ColumnCount-1);
        }
    }
}
