using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace StromaDetectionRBM
{
    class RBMTrainer
    {
        public static void trainRBM(RBM rbm, IRBMInput input, float learningRate, int epochs, int saveInterval, String saveDir, String rbmName)
        {
            int repeat = epochs / saveInterval;
            for (int i = 0; i < repeat; ++i)
            {
                for (int j = 0; j < saveInterval; ++j)
                {
                    float error = rbm.train(input.generateInput(), learningRate);
                    Console.WriteLine(rbmName + ": " + error);
                }

                Matrix<float> weights = rbm.getWeights();
                String outputFile = saveDir + "\\" + rbmName + "_" + repeat + ".weights";
                WeightsHelper.saveWeights(weights, outputFile);
            }
        }

        public interface IRBMInput
        {
            public Matrix<float> generateInput();
        }
    }
}
