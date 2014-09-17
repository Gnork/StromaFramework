using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

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

                    if (r > 0.8f && g > 0.8f && b > 0.8f)
                    {
                        //Console.WriteLine(pos + ": " + r + ", " + g + ", " + b);
                        whiteCount++;
                    }
                }
            }

            if (whiteCount / (width * height) > 0.5)
            {
                return null;
            }

            return result;
        }
    }
}
