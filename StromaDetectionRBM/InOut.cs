﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using VMscope.VirtualSlideAccess;
using VMscope.InteropCore.ImageStreaming;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace StromaDetectionRBM
{
    public class InOut
    {
        private String inputFile;
        private String outputFile;
        private Matrix<float> rbm0Weights;
        private Matrix<float> rbm1Weights;
        private Matrix<float> rbm2Weights;

        private LinkedList<ParseObject> objects;

        public InOut(String[] args)
        {
            if (args.Length < 2)
            {
                error("not enough arguments");
            }

            objects = new LinkedList<ParseObject>();

            this.inputFile = args[0];
            this.outputFile = args[1];

            String[] inputFileSplit = this.inputFile.Split('.');

            if (inputFileSplit[inputFileSplit.Length - 1] == "csv")
            {
                using (var file = System.IO.File.OpenText(inputFile))
                {
                    while (!file.EndOfStream)
                    {
                        String banana = file.ReadLine();

                        if (banana.Length < 1) continue; // skip empty line

                        String[] bananaSplit = banana.Split(';');

                        String id = bananaSplit[0];

                        if (id.ToLower().Equals("id")) continue; //skip headline

                        String filePath = bananaSplit[1];
                        if (!System.IO.Path.IsPathRooted(filePath))
                        {
                            String dirName = System.IO.Path.GetDirectoryName(inputFile);
                            filePath = dirName + "/" + filePath;
                        }
                        IStreamingImage image = Sdk.GetImage(filePath);
                        int upperLeftX = Int32.Parse(bananaSplit[2]);
                        int upperLeftY = Int32.Parse(bananaSplit[3]);
                        int lowerRightX = Int32.Parse(bananaSplit[4]);
                        int lowerRightY = Int32.Parse(bananaSplit[5]);

                        ISRect rect = new ISRect(upperLeftX, upperLeftY, lowerRightX - upperLeftX, lowerRightY - upperLeftY);
                        Bitmap part = image.GetImagePart(rect);

                        objects.AddLast(new ParseObject(id, part));
                    }
                }
            }
            else if(inputFileSplit[inputFileSplit.Length - 1] == "png")
            {
                Bitmap image = new Bitmap(inputFile);

                objects.AddLast(new ParseObject("0", image));
            }
            else
            {
                error("input file must be either csv or png");
            }

            String exeDir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().GetName().CodeBase);

            String rbm0WeightsFile = null;
            String rbm1WeightsFile = null;
            String rbm2WeightsFile = null;

            if (args.Length > 4)
            {
                rbm0WeightsFile = args[2];
                rbm1WeightsFile = args[3];
                rbm2WeightsFile = args[4];
                
            }
            else if (File.Exists(exeDir + "/rbm0.weights") && File.Exists(exeDir + "/rbm1.weights") && File.Exists(exeDir + "/rbm2.weights"))
            {
                rbm0WeightsFile = exeDir + "/rbm0.weights";
                rbm1WeightsFile = exeDir + "/rbm1.weights";
                rbm2WeightsFile = exeDir + "/rbm2.weights";
            }
            else if (File.Exists("rbm0.weights") && File.Exists("rbm1.weights") && File.Exists("rbm2.weights"))
            {
                rbm0WeightsFile = "rbm0.weights";
                rbm1WeightsFile = "rbm1.weights";
                rbm2WeightsFile = "rbm2.weights";
            }
            else
            {
                error("All three RBM weights files must either be given as command line arguments 2, 3, 4 OR must exist in application dir OR must exist in current dir");
            }

            rbm0Weights = WeightsHelper.loadWeights(rbm0WeightsFile);
            rbm1Weights = WeightsHelper.loadWeights(rbm1WeightsFile);
            rbm2Weights = WeightsHelper.loadWeights(rbm2WeightsFile);
        }

        private void error(String message)
        {
            Console.WriteLine("ERROR: " + message + "!");
            Console.WriteLine("");
            Console.WriteLine("Press key to exit...");
            Console.ReadKey();
            Environment.Exit(1);
        }

        public void writeOuput()
        {
            using (System.IO.StreamWriter file = new System.IO.StreamWriter(@outputFile, false))
            {
                file.WriteLine(ParseObject.headline());

                foreach (ParseObject o in objects)
                {
                    file.WriteLine(o.toString());
                }
            }
        }

        public void exportImages()
        {
            foreach (ParseObject o in objects)
            {
                o.getImage().Save(this.outputFile + "." + o.getId() + ".png", ImageFormat.Png);
            }
        }

        public LinkedList<ParseObject> getParseObjects()
        {
            return this.objects;
        }

        public Matrix<float> getRBM0Weights()
        {
            return rbm0Weights;
        }

        public Matrix<float> getRBM1Weights()
        {
            return rbm1Weights;
        }

        public Matrix<float> getRBM2Weights()
        {
            return rbm2Weights;
        }
    }

    public class ParseObject
    {
        private String id;
        private Bitmap image;
        private Boolean stroma;
        private float stromaRatio;

        public ParseObject(String id, Bitmap image)
        {
            this.id = id;
            this.image = image;
        }

        public void setStroma(Boolean isStroma)
        {
            this.stroma = isStroma;
        }

        public void setStromaRatio(float stromaPercentage)
        {
            this.stromaRatio = stromaPercentage;
        }

        public static String headline()
        {
            return "id;stroma;stromaRatio";
        }

        public String toString()
        {
            String isStroma = this.stroma ? "ja" : "nein";
            return id + ";" + isStroma + ";" + stromaRatio;
        }

        public String getId()
        {
            return this.id;
        }

        public Bitmap getImage()
        {
            return this.image;
        }
    }
}
