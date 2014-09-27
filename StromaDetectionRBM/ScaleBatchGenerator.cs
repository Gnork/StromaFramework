using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using System.Drawing;

namespace StromaDetectionRBM
{
    class ScaleBatchGenerator: IBatchGenerator
    {
        private String imageType = "*.png";

        private FileInfo[] positiveSamplesFiles;
        private FileInfo[] negativeSamplesFiles;

        int posCount = 0;
        int negCount = 0;

        public ScaleBatchGenerator(String positiveSamplesPath, String negativeSamplesPath)
        {
            DirectoryInfo positiveSamplesDir = new DirectoryInfo(positiveSamplesPath);
            DirectoryInfo negativeSamplesDir = new DirectoryInfo(negativeSamplesPath);

            positiveSamplesFiles = positiveSamplesDir.GetFiles(imageType);
            negativeSamplesFiles = negativeSamplesDir.GetFiles(imageType);
        }

        public Matrix<float> nextBatch(int batchSize, int patchWidth, int patchHeight)
        {
            int numOfPositive = batchSize / 2;
            float whiteThreshold = 0.5f;

            Matrix<float> batch = Matrix<float>.Build.Dense(batchSize, patchWidth * patchHeight * 3 + 1);

            int i = 0;

            for (; i < numOfPositive; ++i)
            {
                Bitmap image = new Bitmap(positiveSamplesFiles[posCount].FullName);
                posCount = (posCount + 1) % positiveSamplesFiles.Length;
                float[] patchPixels = ImageHelper.generateScaledPatch(image, patchWidth, patchHeight, whiteThreshold);
                if (patchPixels == null)
                {
                    --i;
                    continue;
                }
                batch.SetRow(i, patchPixels);
            }

            for (; i < batchSize; ++i)
            {
                Bitmap image = new Bitmap(negativeSamplesFiles[negCount].FullName);
                negCount = (negCount + 1) % negativeSamplesFiles.Length;
                float[] patchPixels = ImageHelper.generateScaledPatch(image, patchWidth, patchHeight, whiteThreshold);
                if (patchPixels == null)
                {
                    --i;
                    continue;
                }
                batch.SetRow(i, patchPixels);
            }

            return batch;
        }
    }
}
