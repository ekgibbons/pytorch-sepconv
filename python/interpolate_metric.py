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

    sliceExtent = 1000

    ssimFt = np.zeros(sliceExtent)
    ssimBn = np.zeros(sliceExtent)

    psnrFt = np.zeros(sliceExtent)
    psnrBn = np.zeros(sliceExtent)

    nrmseFt = np.zeros(sliceExtent)
    nrmseBn = np.zeros(sliceExtent)

    for sliceNumber in range(sliceExtent):
    
        print("slice: %i/%i" % (sliceNumber,sliceExtent))

        im1 = X[sliceNumber,:,:,0].squeeze()
        im2 = X[sliceNumber,:,:,1].squeeze()
        
        ySepConvFt = SepConvInterpolate.SepConv(im1,im2,"ft")
        ySepConvBn = SepConvInterpolate.SepConv(im1,im2,"bn")
        yLinear = Linear(im1,im2)
        yTrue = y[sliceNumber,:,:,0].astype(float)
    
        ssimFt[sliceNumber] = measure.compare_ssim(ySepConvFt,yTrue)
        ssimBn[sliceNumber] = measure.compare_ssim(ySepConvBn,yTrue)

        psnrFt[sliceNumber] = measure.compare_psnr(ySepConvFt,yTrue)
        psnrBn[sliceNumber] = measure.compare_psnr(ySepConvBn,yTrue)

        nrmseFt[sliceNumber] = measure.compare_nrmse(ySepConvFt,yTrue)
        nrmseBn[sliceNumber] = measure.compare_nrmse(ySepConvBn,yTrue)

    print("SSIM")
    print("\tBN16 -- mean: %f std: %f" % (np.mean(ssimFt),np.std(ssimFt)))
    print("\tBN64 -- mean: %f std: %f" % (np.mean(ssimBn),np.std(ssimBn)))
    print("\n")

    print("PSNR")
    print("\tBN16 -- mean: %f std: %f" % (np.mean(psnrFt),np.std(psnrFt)))
    print("\tBN64 -- mean: %f std: %f" % (np.mean(psnrBn),np.std(psnrBn)))
    print("\n")

    print("NRMSE")
    print("\tBN16 -- mean: %f std: %f" % (np.mean(nrmseFt),np.std(nrmseFt)))
    print("\tBN64 -- mean: %f std: %f" % (np.mean(nrmseBn),np.std(nrmseBn)))
    print("\n")


    plt.show()
    
if __name__ == "__main__":
    main()

