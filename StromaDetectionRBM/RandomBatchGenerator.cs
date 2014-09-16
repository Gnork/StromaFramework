using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex32;
using System.Drawing;
using System.IO;

namespace StromaDetectionRBM
{
    class RandomBatchGenerator: IBatchGenerator
    {
        private String imageType = "*.png";
        private Random random = new Random();


        private FileInfo[] positiveSamplesFiles;
        private FileInfo[] negativeSamplesFiles;

        public RandomBatchGenerator(String positiveSamplesPath, String negativeSamplesPath)
        {
            DirectoryInfo positiveSamplesDir = new DirectoryInfo(positiveSamplesPath);
            DirectoryInfo negativeSamplesDir = new DirectoryInfo(negativeSamplesPath);

            positiveSamplesFiles = positiveSamplesDir.GetFiles(imageType);
            negativeSamplesFiles = negativeSamplesDir.GetFiles(imageType);
        }

        public Matrix<float> nextBatch(int batchSize, int patchWidth, int patchHeight)
        {
            int numOfPositive = batchSize / 2;
            int numOfNegative = batchSize - numOfPositive;

            Matrix<float> batch = Matrix<float>.Build.Dense(batchSize, patchWidth * patchHeight);

            for (int i = 0; i < numOfPositive; ++i)
            {
                int r = random.Next(0, positiveSamplesFiles.Length);
                Bitmap image = new Bitmap(positiveSamplesFiles[r].FullName);
                int x = random.Next(0, image.Width - patchWidth);
                int y = random.Next(0, image.Height - patchHeight);
                float[] patchPixels = ImageHelper.generatePatch(image, x, y, patchWidth, patchHeight);
                batch.SetRow(i, patchPixels);
            }

            for (int i = numOfPositive; i < batchSize; ++i)
            {
                int r = random.Next(0, negativeSamplesFiles.Length);
                Bitmap image = new Bitmap(negativeSamplesFiles[r].FullName);
                int x = random.Next(0, image.Width - patchWidth);
                int y = random.Next(0, image.Height - patchHeight);
                float[] patchPixels = ImageHelper.generatePatch(image, x, y, patchWidth, patchHeight);
                batch.SetRow(i, patchPixels);
            }

            return batch;
        }
    }
}
