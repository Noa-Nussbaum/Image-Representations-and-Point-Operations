# Image Processing
## Image-Representations-and-Point-Operations

Our first Image processing course project


[Here is a link to the assignment](https://github.com/Noa-Nussbaum/Image-Representations-and-Point-Operations/files/8372134/Ex1.pdf)


Exercise 1: Image Representations and Point Operations


Python version: 3.8

I am submitting 6 photos, 1 PDF and 3 Python files;
- ex1_main: one may run the programI've written by running this file.
- ex1_utils: this file contains several functions with which one can edit photos as described below.
- gamma: contains a gamma correction function.

_ex1_utils functions_:

* imReadAndConvert: reads a given image file and converts it into a given representation.
* imDisplay: reads an image as RGB or GRAY_SCALE and displays it in a given representation, utilizing imReadAndConvert .
* transformRGB2YIQ: converts an RGB image to YIQ color space.
* transformYIQ2RGB: converts an YIQ image to RGB color space.
* hsitogramEqualize: performs histogram equalization of a given grayscale or RGB image. The function also displays the input and the equalized output image.
* quantizeImage: performs optimal quantization of a given grayscale or RGB image.

_gamma functions_:
* printGammaValues: prints the gamma values as one changes them.
* gammaDisplay: performs gamma correction on an image with a given γ.
