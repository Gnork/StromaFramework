using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace StromaDetectionRBM
{
    interface IBatchGenerator
    {
        Matrix<float> nextBatch(int batchSize, int patchWidth, int patchHeight);
    }
}
