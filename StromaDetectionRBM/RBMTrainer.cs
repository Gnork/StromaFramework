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

        public static float reconstructionError(Matrix<float> original, Matrix<float> reconstruction)
        {
            float error = 0.0f;

            for (int row = 0; row < original.RowCount; ++row)
            {
                for (int column = 0; column < original.ColumnCount; ++column)
                {
                    error += Math.Abs(original.At(row, column) - reconstruction.At(row, column));
                }
            }

            error /= original.RowCount * original.ColumnCount;

            return error;
        }

        public static float predictionQuality(Matrix<float> matrix)
        {
            int batchSize = matrix.RowCount;
            int numOfPositive = batchSize / 2;

            int corretClassified = 0;
            int cc = matrix.ColumnCount - 1;

            int i = 0;

            for (; i < numOfPositive; ++i)
            {
                if (matrix.At(i, cc) > 0.5f) ++corretClassified;
                Console.WriteLine("label " + i + ": " + matrix.At(i, cc));
            }

            for (; i < batchSize; ++i)
            {
                if (matrix.At(i, cc) < 0.5f) ++corretClassified;
                Console.WriteLine("label " + i + ": " + matrix.At(i, cc));
            }

            return (float)(corretClassified) / (float)(batchSize);
        }

        public interface IRBMInput
        {
            void generateInput();
            Matrix<float> getInput();
        }
    }
}
