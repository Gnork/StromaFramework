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
            String positiveSamplesPath = "";
            String negativeSamplesPath = "";
            String rbm0WeightsPath = "";
            String rbm0SavePath = "";

            int batchSize = 100;
            int patchWidth = 16;
            int patchHeight = 16;

            Random random = new Random();

            IBatchGenerator generator = new RandomBatchGenerator(positiveSamplesPath, negativeSamplesPath);
            Matrix<float> rbm0Weights = WeightsHelper.generateWeights(random);
            RBM rbm0 = new RBM(rbm0Weights, false);
            RBMTrainer.IRBMInput rbm0Input = new RBM0Input(generator, batchSize, patchWidth, patchHeight);

            RBMTrainer.trainRBM(rbm0, rbm0Input, 0.1f, 10000, 100, rbm0SavePath, "RBM0");
        }

        private class RBM0Input: RBMTrainer.IRBMInput{
            private IBatchGenerator generator;

            private int batchSize, patchWidth, patchHeight;

            public RBM0Input(IBatchGenerator generator, int batchSize, int patchWidth, int patchHeight)
            {
                this.generator = generator;
                this.batchSize = batchSize;
                this.patchHeight = patchWidth;
                this.patchWidth = patchWidth;
            }

            public Matrix<float> generateInput()
            {
                return generator.nextBatch(batchSize, patchWidth, patchHeight);
            }
        }   
    }
}
