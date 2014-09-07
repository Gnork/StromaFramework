using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using VMscope.VirtualSlideAccess;
using VMscope.InteropCore.ImageStreaming;
using System.Drawing;

namespace StromaFramework
{
    class Program
    {
        static void Main(string[] args)
        {
            IStreamingImage image = Sdk.GetImage("D:\\mip_projekt\\S-114-HE\\S-114-HE.vsf");
            Bitmap labels = image.GetLabelImage();
            Bitmap tissue = image.GetTissueImage();
            Console.WriteLine("done");
        }
    }
}
