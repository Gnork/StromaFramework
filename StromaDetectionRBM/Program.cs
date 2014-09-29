using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;
using System.IO;

namespace StromaDetectionRBM
{
    class Program
    {
        private static int patchWidth = 64;
        private static int patchHeight = 64;

        private static int scaleWidth = 16;
        private static int scaleHeight = 16;

        private static int scanIncrement = 32;

        private static float classificationThreshold = 0.5f;
        private static float whiteThreshold = 0.6f;

        static void Main(string[] args)
        {
            InOut io = new InOut(args);

            RBM rbm0 = new RBM(io.getRBM0Weights(), false);
            RBM rbm1 = new RBM(io.getRBM1Weights(), false);
            RBM rbm2 = new RBM(io.getRBM2Weights(), false);

            LinkedList<ParseObject> objects = io.getParseObjects();

            foreach (ParseObject o in objects)
            {
                classifyImage(o, rbm0, rbm1, rbm2);
            }

            io.writeOuput();
        }

        private static void classifyImage(ParseObject o, RBM rbm0, RBM rbm1, RBM rbm2)
        {
            Bitmap image = o.getImage();
            LinkedList<float[]> scaledPatches = new LinkedList<float[]>();

            int classWhite = 0;
            int classStroma = 0;
            int classNotStroma = 0;

            for (int y = 0; y < image.Height - patchHeight; y += scanIncrement)
            {
                for (int x = 0; x < image.Width - patchWidth; x += scanIncrement)
                {
                    Bitmap subImage = image.Clone(new Rectangle(x, y, patchWidth, patchHeight), image.PixelFormat);
                    float[] scaledPatch = ImageHelper.generateScaledPatch(subImage, scaleWidth, scaleHeight, whiteThreshold);
                    if (scaledPatch == null)
                    {
                        ++classWhite;
                        continue;
                    }

                    scaledPatches.AddLast(scaledPatch);
                }
            }

            int columnCount = scaleWidth * scaleHeight * 3 + 1;
            Matrix<float> batch = Matrix<float>.Build.Dense(scaledPatches.Count, columnCount);

            int row = 0;
            foreach (float[] scaledPatch in scaledPatches)
            {
                batch.SetRow(row++, scaledPatch);
            }

            Matrix<float> rbm0Hidden = rbm0.getHidden(batch);
            Matrix<float> rbm1Hidden = rbm1.getHidden(rbm0Hidden);
            Matrix<float> rbm1HiddenWithEmptyLabels = MatrixHelper.addEmptyLabels(rbm1Hidden);
            Matrix<float> rbm2Hidden = rbm2.getHidden(rbm1HiddenWithEmptyLabels);

            Matrix<float> rbm2Visible = rbm2.getVisible(rbm2Hidden);

            int lastColumn = rbm2Visible.ColumnCount - 1;

            for (int i = 0; i < rbm2Visible.RowCount; ++i)
            {
                if (rbm2Visible.At(i, lastColumn) > 0.5f) ++classStroma;
                else ++classNotStroma;
            }

            float stroma = classStroma / (float)(classNotStroma + classWhite + classStroma);
            Boolean isStroma = stroma > classificationThreshold;

            Console.WriteLine("Is Stroma: " + isStroma + ", " + stroma);
            Console.WriteLine("Stroma: " + classStroma + ", NotStroma: " + classNotStroma + ", White: " + classWhite);

            o.setStroma(isStroma);
            o.setStromaPercentage(stroma);
        }
    }
}
