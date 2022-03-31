"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import cv2
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
matrix = np.array([[0.299,0.587,0.114],[0.596,-0.275,-0.321],[0.212,-0.523,0.311]])

def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 206664278

def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """

    #read the image
    src = cv2.imread(filename)

    #check whether the image is rgb or grayscale
    if representation == 1: #gray
        if len(src.shape)<2:
            image=src
        elif len(src.shape)==3:
            image = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    if representation == 2: #rgb
        image = src

    #normalize the image to [0,1]
    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return norm_image

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    #transform accordingly
    image = imReadAndConvert(filename,representation)
    plt.imshow(image)
    plt.show()

def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    #image multiplication with transposed matrix
    answer = imgRGB @ matrix.transpose()
    return answer

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    #image multiplication with inversed, transposed matrix
    answer = imgYIQ @ np.linalg.inv(matrix).transpose()
    return answer

def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """
            Equalizes the histogram of an image
            :param imgOrig: Original Histogram
            :ret
        """
        # check whether the image is rgb or grayscale
        isRGB = False
        if len(imgOrig.shape) == 3:
            isRGB = True
            #transform to YIQ
            YIQ = transformRGB2YIQ(imgOrig)
            imgOrig = YIQ[:, :, 0]

        imgOrigInt = (imgOrig * 255).astype("uint8")
        #original image histogram
        histOrig, _ = np.histogram(imgOrigInt.flatten(), 256, range=(0, 255))
        #find original image cumsum
        cumsumOrig = np.cumsum(histOrig)
        #normalize cumulative histogram
        cdfNorm = (cumsumOrig * 255 / cumsumOrig[-1]).astype("uint8")
        #transform
        imgEq = cdfNorm[imgOrigInt]
        #check bounds
        imgEq[imgEq < 0] = 0
        imgEq[imgEq > 255] = 255
        #find new image cumsum
        histEq, _ = np.histogram(imgEq.flatten(), 256, range=(0, 255))

        #normalize the image to [0,1]
        imgEq = (imgEq / 255)

        #if RGB
        if isRGB == True:
            #modify y channel
            YIQ[:, :, 0] = imgEq.copy()
            #transform back to RGB
            imgEq = transformYIQ2RGB(YIQ)
            return imgEq, histOrig, histEq

        return imgEq, histOrig, histEq

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    #if RGB image
    isRGB = False
    if len(imOrig.shape) == 3:
        isRGB = True
        #transform to YIQ
        imgYIQ = transformRGB2YIQ(imOrig)
        #y-channel
        imOrig = imgYIQ[:, :, 0]

    imgOrigInt = (imOrig * 255).astype("uint8")
    #find original image histogram
    histOrig, _ = np.histogram(imgOrigInt.flatten(), 256, range=(0, 255))

    #mse array
    mseList = []
    #images list
    imagesList = []
    global intensities, z, q

    for j in range(nIter):
        encodeImg = imgOrigInt.copy()
        #find z
        #initiate z
        if j == 0:
            z = np.arange(0, 255 - int(256 / nQuant) + 1, int(256 / nQuant))
            z = np.append(z, 255)
            intensities = np.array(range(256))
        else:
            for r in range(1, len(z) - 2):
                new_z_r = int((q[r - 1] + q[r]) / 2)
                if new_z_r != z[r - 1] and new_z_r != z[r + 1]:
                    z[r] = new_z_r

        #find q
        q = np.array([], dtype=np.float64)
        for i in range(len(z) - 1):
            mask = np.logical_and((z[i] < encodeImg), (encodeImg < z[i + 1]))
            if i is not (len(z) - 2):
                #calculate weighted mean
                if sum(histOrig[z[i]:z[i + 1]]) != 0:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1]], weights=histOrig[z[i]:z[i + 1]]))
                else:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1]], weights=histOrig[z[i]:z[i + 1]] + 0.001))
                encodeImg[mask] = int(q[i])

            else:
                #calculate weighted mean
                if sum(histOrig[z[i]:z[i + 1]]) != 0:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1] + 1], weights=histOrig[z[i]:z[i + 1] + 1]))
                else:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1] + 1],
                                                weights=histOrig[z[i]:z[i + 1] + 1] + 0.001))
                encodeImg[mask] = int(q[i])

        #find mse
        mseList.append((np.square(np.subtract(imgOrigInt, encodeImg))).mean())
        #normalize the image to [0,1]
        encodeImg = (encodeImg / 255)

        if isRGB:
            #y-channel
            imgYIQ[:, :, 0] = encodeImg.copy()
            #transform back to RGB
            encodeImg = transformYIQ2RGB(imgYIQ)

        imagesList.append(encodeImg)

    return imagesList, mseList

