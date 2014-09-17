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
        public static void trainRBM(RBM rbm, IRBMInput input, float learningRate, int epochs, int saveInterval, String saveDir, String rbmName, int visibleLayer, int hiddenLayer)
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
                    Console.WriteLine(rbmName + "; Epoche: " + (i * saveInterval + j) +  "; Error: " + error);

                    if (error < minError)
                    {
                        minError = error;
                        minWeights = rbm.getWeights();
                    }
                    
                    thread.Join();
                    currentInput = input.getInput();
                }

                // save best weights from last interval
                if(minWeights != null)
                {
                    String outputFile = saveDir + "\\" + rbmName + "_" + visibleLayer + "_" + hiddenLayer + "_" + minError + ".weights";
                    WeightsHelper.saveWeights(minWeights, outputFile);
                    minWeights = null;
                    Console.WriteLine("weights saved");
                }         
            }
        }

        public interface IRBMInput
        {
            void generateInput();
            Matrix<float> getInput();
        }
    }
}
