using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;

namespace StromaDetectionRBM
{
    class ImageHelper
    {
        public static float[] generatePatch(Bitmap image, int x, int y, int width, int height){
            float[] result = new float[width * height * 3 + 1];

            int whiteCount = 0;
            int pos = 0;
            result[pos++] = 1.0f;

            for (int yPos = y; yPos < y + height; ++yPos)
            {
                for (int xPos = x; xPos < x + width; ++xPos)
                {
                    Color c = image.GetPixel(xPos, yPos);
                    float r = c.R / 255.0f;
                    float g = c.G / 255.0f;
                    float b = c.B / 255.0f;
                    result[pos++] = r;
                    result[pos++] = g;
                    result[pos++] = b;

                    // count white pixels
                    if (r > 0.8f && g > 0.8f && b > 0.8f)
                    {
                        whiteCount++;
                    }
                }
            }

            // return null if patch is more than 50% white
            if (whiteCount / (width * height) > 0.5)
            {
                return null;
            }

            return result;
        }

        public static float[] generateScaledPatch(Bitmap image, int scaleWidth, int scaleHeight, float whiteThreshold)
        {
            float[] result = new float[scaleWidth * scaleHeight * 3 + 1];

            int whiteCount = 0;
            int pos = 0;
            result[pos++] = 1.0f;

            Bitmap scaledImage = new Bitmap(image, new Size(scaleWidth, scaleHeight));

            for (int yPos = 0; yPos < scaleHeight; ++yPos)
            {
                for (int xPos = 0; xPos < scaleWidth; ++xPos)
                {
                    Color c = scaledImage.GetPixel(xPos, yPos);
                    float r = c.R / 255.0f;
                    float g = c.G / 255.0f;
                    float b = c.B / 255.0f;
                    result[pos++] = r;
                    result[pos++] = g;
                    result[pos++] = b;

                    // count white pixels
                    if (r > 0.9f && g > 0.9f && b > 0.9f)
                    {
                        whiteCount++;
                    }
                }
            }

            // return null if patch is mostly white
            if (((float)whiteCount) / (scaleWidth * scaleHeight) > whiteThreshold) return null;

            return result;
        }

        public static void persistPatch(float[] patch, int width, int height, String filePath)
        {
            Bitmap image = new Bitmap(width, height);

            int pos = 0;

            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    int r = Math.Max(0, Math.Min(255, (int)(patch[pos++] * 255.0f)));
                    int g = Math.Max(0, Math.Min(255, (int)(patch[pos++] * 255.0f)));
                    int b = Math.Max(0, Math.Min(255, (int)(patch[pos++] * 255.0f)));

                    image.SetPixel(x, y, Color.FromArgb(r, g, b));
                }
            }

            image.Save(filePath);
        }

        public static void persistOriginalAndReconstruction(int width, int height, Matrix<float> original, Matrix<float> reconstruction, String dirPath)
        {
            Matrix<float> originalCropped = original.RemoveColumn(0);
            Matrix<float> reconstructionCropped = reconstruction.RemoveColumn(0);

            for (int row = 0; row < original.RowCount; ++row)
            {
                String originalOutput = dirPath + "\\" + row + "_original.png";
                String reconstructionOutput = dirPath + "\\" + row + "_reconstruction.png";
                persistPatch(originalCropped.Row(row).ToArray(), width, height, originalOutput);
                persistPatch(reconstructionCropped.Row(row).ToArray(), width, height, reconstructionOutput);
            }
        }
    }
}
