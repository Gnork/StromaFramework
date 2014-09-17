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
            String positiveSamplesPath = "D:\\StromaSet\\S-114-HE\\stroma";
            String negativeSamplesPath = "D:\\StromaSet\\S-114-HE\\not-stroma";
            String rbm0WeightsPath = "D:\\StromaSet\\weights\\RBM0_769_350_0,09807873.weights";
            String rbm0SavePath = "D:\\StromaSet\\weights";

            int batchSize = 100;
            int patchWidth = 16;
            int patchHeight = 16;

            Random random = new Random();

            int rbm0Visible = patchWidth * patchHeight * 3 + 1;
            int rbm0Hidden = 350;

            IBatchGenerator generator = new RandomBatchGenerator(positiveSamplesPath, negativeSamplesPath);

            //Matrix<float> rbm0Weights = WeightsHelper.generateWeights(rbm0Visible, rbm0Hidden, random);
            Matrix<float> rbm0Weights = WeightsHelper.loadWeights(rbm0WeightsPath);

            RBM rbm0 = new RBM(rbm0Weights, false);
            RBMTrainer.IRBMInput rbm0Input = new RBM0Input(generator, batchSize, patchWidth, patchHeight);

            RBMTrainer.trainRBM(rbm0, rbm0Input, 0.001f, 10000, 100, rbm0SavePath, "RBM0", rbm0Visible, rbm0Hidden);
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
                this.input = rbm0.getHidden(batch);
            }
        }
    }
}
