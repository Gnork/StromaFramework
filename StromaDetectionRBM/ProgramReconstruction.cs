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
            String positiveSamplesPath = "D:\\StromaSet\\S-114-HE_64\\crossvalidation\\stroma";
            String negativeSamplesPath = "D:\\StromaSet\\S-114-HE_64\\crossvalidation\\not-stroma";
            String outputPath = "D:\\StromaSet\\reconstructions";

            String rbm0WeightsPath = "D:\\StromaSet\\weights\\RBM0_T1_769_500_139_0,04191353.weights";
            String rbm1WeightsPath = "D:\\StromaSet\\weights\\RBM1_T1_500_75_375_0,07180008.weights";
            String rbm2WeightsPath = "D:\\StromaSet\\weights\\RBM2_TOP_T2_76_40_5_0,1721749.weights";

            int batchSize = 100;
            int patchWidth = 16;
            int patchHeight = 16;

            IBatchGenerator generator = new ScaleBatchGenerator(positiveSamplesPath, negativeSamplesPath);

            Matrix<float> rbm0Weights = WeightsHelper.loadWeights(rbm0WeightsPath);
            Matrix<float> rbm1Weights = WeightsHelper.loadWeights(rbm1WeightsPath);
            Matrix<float> rbm2Weights = WeightsHelper.loadWeights(rbm2WeightsPath);

            RBM rbm0 = new RBM(rbm0Weights, false);
            RBM rbm1 = new RBM(rbm1Weights, false);
            RBM rbm2 = new RBM(rbm2Weights, false);

            Matrix<float> batch = generator.nextBatch(batchSize, patchWidth, patchHeight);

            Matrix<float> rbm0Hidden = rbm0.getHidden(batch);
            Matrix<float> rbm1Hidden = rbm1.getHidden(rbm0Hidden);
            Matrix<float> rbm1HiddenWithEmptyLabels = MatrixHelper.addEmptyLabels(rbm1Hidden);
            Matrix<float> rbm2Hidden = rbm2.getHidden(rbm1HiddenWithEmptyLabels);

            Matrix<float> rbm2Visible = rbm2.getVisible(rbm2Hidden);
            Matrix<float> rbm2VisibleWithoutLabels = MatrixHelper.removeLabels(rbm2Visible);
            Matrix<float> rbm1Visible = rbm1.getVisible(rbm2VisibleWithoutLabels);
            Matrix<float> rbm0Visible = rbm0.getVisible(rbm1Visible);

            ImageHelper.persistOriginalAndReconstruction(patchWidth, patchHeight, batch, rbm0Visible, outputPath);

            Console.WriteLine("Image Reconstruction: " + RBMTrainer.reconstructionError(batch, rbm0Visible));
            Console.WriteLine("Prediction Quality: " + RBMTrainer.predictionQuality(rbm2Visible));

            Console.WriteLine("press key to exit: ");
            Console.ReadKey();
        }
    }
}
