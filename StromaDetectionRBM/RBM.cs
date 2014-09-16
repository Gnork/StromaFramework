using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace StromaDetectionRBM
{
    class RBM
    {
        private Matrix<float> weights;
        private Random random;
        private Func<float, float> binarizeHidden;

        public RBM(Matrix<float> weights, bool binarizeHidden)
        {
            this.weights = weights;
            this.random = new Random();
            this.binarizeHidden = identity;
            if (binarizeHidden) this.binarizeHidden = binarize;
        }

        public float train(Matrix<float> data, float learningRate)
        {
            // positive hidden representation
            Matrix<float> hidden = getHidden(data, binarizeHidden);
            // positive associations
            Matrix<float> positiveAssociations = data.TransposeThisAndMultiply(hidden);
            // visible reconstruction
            Matrix<float> visible = getVisible(hidden);
            // negative hidden representation
            hidden = getHidden(visible, binarizeHidden);
            // negative associations
            Matrix<float> negativeAssociations = visible.TransposeThisAndMultiply(hidden);
            // update weights
            this.weights.Add(positiveAssociations.Subtract(negativeAssociations).Multiply(learningRate / data.RowCount));

            return error(data, hidden, visible);
        }

        public float error(Matrix<float> data)
        {
            Matrix<float> hidden = getHidden(data);
            Matrix<float> visible = getVisible(hidden);

            return error(data, hidden, visible);
        }

        public Matrix<float> getHidden(Matrix<float> data)
        {
            return getHidden(data, identity);
        }

        private Matrix<float> getHidden(Matrix<float> data, Func<float, float> f)
        {
            // calculate hidden probabilities or states
            Matrix<float> hidden = data.Multiply(this.weights).Map(logistic, Zeros.Include).Map(f);
            // reset bias
            hidden.SetColumn(0, Vector<float>.Build.Dense(hidden.RowCount, 1.0f));

            return hidden;
        }

        public Matrix<float> getVisible(Matrix<float> hidden)
        {
            // calculate visible reconstructions
            Matrix<float> visible = hidden.TransposeAndMultiply(this.weights).Map(logistic, Zeros.Include);
            // reset bias
            visible.SetColumn(0, Vector<float>.Build.Dense(visible.RowCount, 1.0f));

            return visible;
        }

        private float error(Matrix<float> data, Matrix<float> hidden, Matrix<float> visible)
        {
            // calculate normalized mean squared error
            return (float)(Math.Sqrt(data.Subtract(visible).Map(square).RowSums().Sum() / (data.RowCount * this.weights.RowCount)));
        }

        private float logistic(float value)
        {
            return (float)(1.0 / (1.0 + Math.Exp(-value)));
        }

        private float square(float value)
        {
            return value * value;
        }

        private float binarize(float value)
        {
            return value > (float)random.NextDouble() ? 1.0f : 0.0f;
        }

        private float identity(float value)
        {
            return value;
        }
    }
}
