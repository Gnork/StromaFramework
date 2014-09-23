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
        public static void trainRBM(RBM rbm, IRBMInput input, float learningRate, int epochs, int saveInterval, String saveDir, String trainingName, int visibleLayer, int hiddenLayer)
        {
            input.generateInput();
            Matrix<float> currentInput = input.getInput();

            float minError = float.MaxValue;
            Matrix<float> minWeights = null;
            float error = float.MaxValue;

            int repeat = epochs / saveInterval;
            for (int i = 0; i < repeat; ++i)
            {
                for (int j = 0; j < saveInterval; ++j)
                {
                    Thread thread = new Thread(input.generateInput);
                    thread.Start();

                    error = rbm.train(currentInput, learningRate);
                    Console.WriteLine(trainingName + "; Epoche: " + (i * saveInterval + j) +  "; Error: " + error);

                    if (error < minError)
                    {
                        minError = error;
                        minWeights = rbm.getWeights();
                    }
                    
                    thread.Join();
                    currentInput = input.getInput();
                }

                // save best weights from last interval
                String outputFile = saveDir + "\\" + trainingName + "_" + visibleLayer + "_" + hiddenLayer + "_" + i + "_" + minError + ".weights";
                WeightsHelper.saveWeights(minWeights, outputFile);
                minError = float.MaxValue;
                Console.WriteLine("weights saved");
            }
        }

        private Matrix<float> avgMatrix(Matrix<float>[] matrices)
        {
            Matrix<float> avgMatrix = Matrix<float>.Build.Dense(matrices[0].RowCount, matrices[0].RowCount, 0.0f);

            for (int i = 0; i < matrices.Length; ++i)
            {
                avgMatrix.Add(matrices[i], avgMatrix);
            }
            avgMatrix.Multiply(1.0f / matrices.Length, avgMatrix);

            return avgMatrix;
        }

        public interface IRBMInput
        {
            void generateInput();
            Matrix<float> getInput();
        }
    }
}
