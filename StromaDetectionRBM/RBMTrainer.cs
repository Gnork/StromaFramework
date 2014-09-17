using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.Threading;

namespace StromaDetectionRBM
{
    class RBMTrainer
    {
        public static void trainRBM(RBM rbm, IRBMInput input, float learningRate, int epochs, int saveInterval, String saveDir, String rbmName)
        {
            input.generateInput();
            Matrix<float> currentInput = input.getInput();

            int repeat = epochs / saveInterval;
            for (int i = 0; i < repeat; ++i)
            {
                for (int j = 0; j < saveInterval; ++j)
                {
                    Thread thread = new Thread(input.generateInput);
                    thread.Start();

                    float error = rbm.train(currentInput, learningRate);
                    Console.WriteLine(rbmName + ": " + error);

                    thread.Join();
                    Matrix<float> nextInput = input.getInput();
                }

                Matrix<float> weights = rbm.getWeights();
                String outputFile = saveDir + "\\" + rbmName + "_" + repeat + ".weights";
                WeightsHelper.saveWeights(weights, outputFile);
            }
        }

        public interface IRBMInput
        {
            void generateInput();
            Matrix<float> getInput();
        }
    }
}
