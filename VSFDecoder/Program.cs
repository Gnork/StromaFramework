using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using VMscope.VirtualSlideAccess;
using VMscope.InteropCore.ImageStreaming;
using System.Drawing;
using System.Drawing.Imaging;

namespace VSFDecoder
{
    class Program
    {
        static void Main(string[] args)
        {
            IStreamingImage image = Sdk.GetImage(@"D:\mip_projekt\S-114-HE\S-114-HE.vsf");

            Size size = image.Size;
            int levels = image.Levels;
            int patchSize = 64;

            int maxWidth = size.Width - patchSize;
            int maxHeight = size.Height - patchSize;

            for (int y = 0; y < maxHeight; y += patchSize)
            {
                for (int x = 0; x < maxWidth; x += patchSize)
                {
                    ISRect rect = new ISRect(x, y, patchSize, patchSize);
                    Bitmap part = image.GetImagePart(rect);
                    if (checkColor(part))
                    {
                        part.Save("D:\\StromaSet\\S-114-HE_64\\parts\\" + x + "_" + y + ".png", ImageFormat.Png);
                        Console.WriteLine("SAVE: " + x + " ; " + y);
                    }
                    else
                    {
                        Console.WriteLine(x + " ; " + y);
                    }

                }
            }

            Console.WriteLine("press key to exit: ");
            Console.ReadKey();
        }


        private static bool checkColor(Bitmap image)
        {
            float numOfWhite = 0;
            for (int j = 0; j < image.Height; ++j)
            {
                for (int i = 0; i < image.Width; ++i)
                {
                    Color pixel = image.GetPixel(i, j);
                    byte r = pixel.R;
                    if (pixel.R > 220 && pixel.G > 220 && pixel.B > 220)
                    {
                        numOfWhite++;
                    }
                }
            }
            if (numOfWhite / (image.Size.Width * image.Size.Height) < 0.6)
            {
                return true;
            }
            return false;
        }
    }
}
