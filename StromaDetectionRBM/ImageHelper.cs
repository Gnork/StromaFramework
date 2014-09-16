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
            float[] result = new float[width * height * 3];

            int pos = 0;

            for (int yPos = y; yPos < y + height; ++yPos)
            {
                for (int xPos = x; xPos < x + width; ++xPos)
                {
                    Color c = image.GetPixel(xPos, yPos);
                    result[pos++] = c.R / 255.0f;
                    result[pos++] = c.G / 255.0f;
                    result[pos++] = c.B / 255.0f;
                }
            }

            return result;
        }
    }
}
