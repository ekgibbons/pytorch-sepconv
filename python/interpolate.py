import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"       

import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from scipy import stats

import GenerateHeartData
import SepConvInterpolate

def Linear(im1, im2):
    """
    """
    
    outputImage = (im1 + im2)/2

    return outputImage.astype(float)

def main():
    """
    """

    pathRead = "/v/raid1a/egibbons/data/deep-slice"
    data = np.load("%s/training_hearts_00.npy" % (pathRead))

    X, y = GenerateHeartData.DataGenerator(data)

    sliceNumber = 1000
    
    im1 = X[sliceNumber,:,:,0].squeeze()
    im2 = X[sliceNumber,:,:,1].squeeze()
    
    ySepConvFt = SepConvInterpolate.SepConv(im1,im2,"ft")
    yLinear = Linear(im1,im2)
    yTrue = y[sliceNumber,:,:,0].astype(float)
    
    plt.figure()
    plt.imshow(ySepConvFt,cmap="gray")

    plt.figure()
    plt.imshow(yLinear,cmap="gray")
    
    plt.figure()
    plt.imshow(yTrue,cmap="gray")
    
    montage = np.concatenate((im1,ySepConvFt,im2),axis=1)

    plt.figure()
    plt.imshow(montage,cmap="gray")


    montage = np.concatenate((im1,yTrue,im2),axis=1)

    plt.figure()
    plt.imshow(montage,cmap="gray")


    plt.show()
    
if __name__ == "__main__":
    main()

