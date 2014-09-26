using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace StromaDetectionRBM
{
    class ProgramTraining
    {
        static void Main(string[] args)
        {
            String positiveSamplesPath = "D:\\StromaSet\\S-114-HE_64\\training\\stroma";
            String negativeSamplesPath = "D:\\StromaSet\\S-114-HE_64\\training\\not-stroma";
            String rbmSavePath = "D:\\StromaSet\\weights";

            String rbm0WeightsPath = "D:\\StromaSet\\weights\\RBM0_T1_769_500_139_0,04191353.weights";
            String rbm1WeightsPath = "D:\\StromaSet\\weights\\RBM1_T1_500_75_375_0,07180008.weights";
            String rbm2WeightsPath = "D:\\StromaSet\\weights\\RBM2_TOP_T2_76_40_5_0,1721749.weights";

            int batchSize = 100;
            int patchWidth = 16;
            int patchHeight = 16;

            Random random = new Random();

            int rbm0Visible = patchWidth * patchHeight * 3 + 1;
            int rbm0Hidden = 500;

            int rbm1Visible = rbm0Hidden;
            int rbm1Hidden = 75;

            int rbm2Visible = rbm1Hidden + 1;
            int rbm2Hidden = 40;

            IBatchGenerator generator = new ScaleBatchGenerator(positiveSamplesPath, negativeSamplesPath);

            //Matrix<float> rbm0Weights = WeightsHelper.generateWeights(rbm0Visible, rbm0Hidden, random);
            Matrix<float> rbm0Weights = WeightsHelper.loadWeights(rbm0WeightsPath);
            //Matrix<float> rbm1Weights = WeightsHelper.generateWeights(rbm1Visible, rbm1Hidden, random);
            Matrix<float> rbm1Weights = WeightsHelper.loadWeights(rbm1WeightsPath);
            //Matrix<float> rbm2Weights = WeightsHelper.generateWeights(rbm2Visible, rbm2Hidden, random);
            Matrix<float> rbm2Weights = WeightsHelper.loadWeights(rbm2WeightsPath);

            RBM rbm0 = new RBM(rbm0Weights, false);
            RBM rbm1 = new RBM(rbm1Weights, false);
            RBM rbm2 = new RBM(rbm2Weights, false);

            //RBMTrainer.IRBMInput rbm0Input = new RBM0Input(generator, batchSize, patchWidth, patchHeight);
            //RBMTrainer.trainRBM(rbm0, rbm0Input, 0.01f, 100000, 100, rbmSavePath, "RBM0_LABEL_T1", rbm0Visible, rbm0Hidden);

            //RBMTrainer.IRBMInput rbm1Input = new RBM1Input(generator, batchSize, patchWidth, patchHeight, rbm0);
            //RBMTrainer.trainRBM(rbm1, rbm1Input, 0.01f, 2000000, 1000, rbmSavePath, "RBM1_LABELS_T2", rbm1Visible, rbm1Hidden);

            RBMTrainer.IRBMInput rbm2Input = new RBM2Input(generator, batchSize, patchWidth, patchHeight, rbm0, rbm1);
            RBMTrainer.trainRBM(rbm2, rbm2Input, 0.03f, 1000000, 1000, rbmSavePath, "RBM2_TOP_T3", rbm2Visible, rbm2Hidden);
        }

        private class RBM0Input: RBMTrainer.IRBMInput{
            private IBatchGenerator generator;

            private int batchSize, patchWidth, patchHeight;
            private Matrix<float> input = null;

            public RBM0Input(IBatchGenerator generator, int batchSize, int patchWidth, int patchHeight)
            {
                this.generator = generator;
                this.batchSize = batchSize;
                this.patchHeight = patchWidth;
                this.patchWidth = patchWidth;
            }

            public Matrix<float> getInput()
            {
                return input;
            }

            public void generateInput()
            {
                this.input = generator.nextBatch(batchSize, patchWidth, patchHeight);
            }
        }

        private class RBM1Input : RBMTrainer.IRBMInput
        {
            private IBatchGenerator generator;

            private int batchSize, patchWidth, patchHeight;
            private Matrix<float> input;
            RBM rbm0;

            public RBM1Input(IBatchGenerator generator, int batchSize, int patchWidth, int patchHeight, RBM rbm0)
            {
                this.generator = generator;
                this.batchSize = batchSize;
                this.patchHeight = patchWidth;
                this.patchWidth = patchWidth;
                this.rbm0 = rbm0;
            }

            public Matrix<float> getInput()
            {
                return input;
            }

            public void generateInput()
            {
                Matrix<float> batch = generator.nextBatch(batchSize, patchWidth, patchHeight);
                Matrix<float> rbm0Hidden = rbm0.getHidden(batch);
                this.input = rbm0Hidden;
            }
        }

        private class RBM2Input : RBMTrainer.IRBMInput
        {
            private IBatchGenerator generator;

            private int batchSize, patchWidth, patchHeight;
            private Matrix<float> input;
            RBM rbm0, rbm1;

            public RBM2Input(IBatchGenerator generator, int batchSize, int patchWidth, int patchHeight, RBM rbm0, RBM rbm1)
            {
                this.generator = generator;
                this.batchSize = batchSize;
                this.patchHeight = patchWidth;
                this.patchWidth = patchWidth;
                this.rbm0 = rbm0;
                this.rbm1 = rbm1;
            }

            public Matrix<float> getInput()
            {
                return input;
            }

            public void generateInput()
            {
                Matrix<float> batch = generator.nextBatch(batchSize, patchWidth, patchHeight);
                Matrix<float> rbm0Hidden = rbm0.getHidden(batch);
                Matrix<float> rbm1Hidden = rbm1.getHidden(rbm0Hidden);
                this.input = MatrixHelper.addLabels(rbm1Hidden);
            }
        }
    }
}
