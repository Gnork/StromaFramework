StromaFramework
===============

### usage

```
StromaDetection.exe input.csv output.csv [rbm0.weights rbm1.weights rbm2.weights]
StromaDetection.exe input.png output.csv [rbm0.weights rbm1.weights rbm2.weights]

# rbm weights are optional parameters
# weights files must exist in application dir or current dir, if not specified as parameters
```

### output

Beispiel:

```
id;stroma;stromaRatio
0;ja;0.751
1;nein;0.554
2;nein;0.111
```

### classification

This software classifies medical image data and decides whether the image is a Stroma image or not. An image is classified as Stroma, if it contains more than 60% Stroma.

### program entry point
```
StromaDetectionRBM/Program.cs
```
