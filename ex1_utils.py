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
    src = cv2.imread(filename)
    if representation == 1: #gray
        if len(src.shape)<2:
            image=src
        elif len(src.shape)==3:
            image = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    if representation == 2: #rgb
        image = src

    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return norm_image

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image = imReadAndConvert(filename,representation)
    plt.imshow(image)
    plt.show()

def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    answer = imgRGB @ matrix.transpose()
    return answer


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    answer = imgYIQ @ np.linalg.inv(matrix).transpose()
    return answer


# def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
#     """
#         Equalizes the histogram of an image
#         :param imgOrig: Original Histogram
#         :ret
#     """
#     norm_image = cv2.normalize(imgOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     flattenedorg=np.ndarray.flatten(norm_image)
#     histOrg=np.histogram(flattenedorg)
#
#     cumsumarr=np.cumsum(histOrg)
#     LUT=np.ceiling(cumsumarr/cumsumarr.max()*255)
#
#     imgEq = np.zeros_like(imgOrig, dtype=float)
#     for i in range(256):
#         imgEq[flattenedorg == i] = int(LUT[i])
#
#     flattenednew=np.ndarray.flatten(imgEq)
#     histEQ=np.histogram(flattenednew)
#     imgEq = cv2.normalize(imgEq, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#
#     return imgEq,histOrg,histEQ

def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """
            Equalizes the histogram of an image
            :param imgOrig: Original Histogram
            :ret
        """
        # input checking
        if imgOrig is None:
            raise Exception("Error: imgOrig is None!")
        # handle RGB images
        flagRGB = False
        if len(imgOrig.shape) == 3:  # RGB image
            flagRGB = True
            imgYIQ = transformRGB2YIQ(imgOrig)  # transform to YIQ color space
            imgOrig = imgYIQ[:, :, 0]

        imgOrigInt = (imgOrig * 255).astype("uint8")
        # find the histogram of the original image
        histOrig, _ = np.histogram(imgOrigInt.flatten(), 256, range=(0, 255))
        cumsumOrig = np.cumsum(histOrig)  # calculate cumulative-sum of the original image
        cdfNorm = (cumsumOrig * 255 / cumsumOrig[-1]).astype("uint8")  # normalize cumulative histogram
        imgEq = cdfNorm[imgOrigInt]  # apply the transformation
        # bounds checking:
        imgEq[imgEq < 0] = 0
        imgEq[imgEq > 255] = 255
        # calculate cumulative-sum of the new image
        histEq, _ = np.histogram(imgEq.flatten(), 256, range=(0, 255))

        # normalize image into range [0,1]
        imgEq = imgEq / 255

        if flagRGB == True:  # RGB image
            imgYIQ[:, :, 0] = imgEq.copy()  # modify y channel
            imgEq = transformYIQ2RGB(imgYIQ)  # transform back to RGB
            return imgEq, histOrig, histEq

        return imgEq, histOrig, histEq

# def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
#     """
#         Quantized an image in to **nQuant** colors
#         :param imOrig: The original image (RGB or Gray scale)
#         :param nQuant: Number of colors to quantize the image to
#         :param nIter: Number of optimization loops
#         :return: (List[qImage_i],List[error_i])
#     """
#     imageList=[]
#     mseList=[]
#
#     norm_image = cv2.normalize(imOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     flattenedorg = np.ndarray.flatten(norm_image)
#     histOrg = np.histogram(flattenedorg)
#
#     z=[0]
#     q=[]
#     n=255/nQuant
#
#     for i in range(nQuant):
#         z.append(n)
#         n+=n
#
#     return imageList, mseList

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # input checking
    if imOrig is None:
        raise Exception("Error: imOrig is None!")
    if nQuant > 256:
        raise ValueError("nQuant is greater then 256!")
    if nIter < 0:
        raise ValueError("Number of optimization loops must be a positive number!")

    # handle RGB images
    flagRGB = False
    if len(imOrig.shape) == 3:  # RGB image
        flagRGB = True
        imgYIQ = transformRGB2YIQ(imOrig)  # transform to YIQ color space
        imOrig = imgYIQ[:, :, 0]  # y-channel

    imgOrigInt = (imOrig * 255).astype("uint8")
    # find the histogram of the original image
    histOrig, _ = np.histogram(imgOrigInt.flatten(), 256, range=(0, 255))

    errors = []  # errors array
    encodedImages = []  # contains all the encoded images- finally return the minimum error image
    global intensities, z, q

    for j in range(nIter):
        encodeImg = imgOrigInt.copy()
        # Finding z - the values that each of the segments intensities will map to.
        if j is 0:  # first iteration INIT z
            z = np.arange(0, 255 - int(256 / nQuant) + 1, int(256 / nQuant))
            z = np.append(z, 255)
            intensities = np.array(range(256))
        else:  # not the first iteration
            for r in range(1, len(z) - 2):
                new_z_r = int((q[r - 1] + q[r]) / 2)
                if new_z_r != z[r - 1] and new_z_r != z[r + 1]:  # to avoid division by 0
                    z[r] = new_z_r

        # Finding q - the values that each of the segments intensities will map to.
        q = np.array([], dtype=np.float64)
        for i in range(len(z) - 1):
            mask = np.logical_and((z[i] < encodeImg), (encodeImg < z[i + 1]))  # the current cluster
            if i is not (len(z) - 2):
                # calculate weighted mean
                if sum(histOrig[z[i]:z[i + 1]]) != 0:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1]], weights=histOrig[z[i]:z[i + 1]]))
                else:  # to avoid division by 0
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1]], weights=histOrig[z[i]:z[i + 1]] + 0.001))
                encodeImg[mask] = int(q[i])  # apply the changes to the encoded image

            else:  # i is len(z)-2 , add 255
                # calculate weighted mean
                if sum(histOrig[z[i]:z[i + 1]]) != 0:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1] + 1], weights=histOrig[z[i]:z[i + 1] + 1]))
                else:  # to avoid division by 0
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1] + 1],
                                                weights=histOrig[z[i]:z[i + 1] + 1] + 0.001))
                encodeImg[mask] = int(q[i])  # apply the changes on the encoded image

        errors.append((np.square(np.subtract(imgOrigInt, encodeImg))).mean())  # calculate error
        encodeImg = encodeImg / 255  # normalize to range [0,1]

        if flagRGB:  # RGB image
            imgYIQ[:, :, 0] = encodeImg.copy()  # modify y channel
            encodeImg = transformYIQ2RGB(imgYIQ)  # transform back to RGB

        encodedImages.append(encodeImg)

        # checking whether we have come to convergence
        if j > 1 and abs(errors[j - 1] - errors[j]) < 0.01:
            print("we have come to convergence after {} iterations!".format(j + 1))
            break

    return encodedImages, errors

