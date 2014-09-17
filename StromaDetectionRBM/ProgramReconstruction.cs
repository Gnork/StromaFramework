using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace StromaDetectionRBM
{
    class ProgramReconstruction
    {
        static void Main(string[] args)
        {
            String positiveSamplesPath = "D:\\StromaSet\\S-114-HE\\stroma";
            String negativeSamplesPath = "D:\\StromaSet\\S-114-HE\\not-stroma";
            String rbm0WeightsPath = "D:\\StromaSet\\weights\\RBM0_769_350_0,09050716.weights";
            String outputPath = "D:\\StromaSet\\reconstructions";

            int batchSize = 100;
            int patchWidth = 16;
            int patchHeight = 16;

            Random random = new Random();

            IBatchGenerator generator = new RandomBatchGenerator(positiveSamplesPath, negativeSamplesPath);

            Matrix<float> rbm0Weights = WeightsHelper.loadWeights(rbm0WeightsPath);

            RBM rbm0 = new RBM(rbm0Weights, false);

            Matrix<float> batch = generator.nextBatch(batchSize, patchWidth, patchHeight);

            Matrix<float> rbm0Hidden = rbm0.getHidden(batch);
            Matrix<float> rbm0Visible = rbm0.getVisible(rbm0Hidden);

            ImageHelper.persistOriginalAndReconstruction(patchWidth, patchHeight, batch, rbm0Visible, outputPath);

        }
    }
}
