import glob
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"       

import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from scipy import stats

import GenerateHeartData
import SepConvInterpolate
from matlab import  matreader

def Linear(im1, im2):
    """
    """
    
    outputImage = (im1 + im2)/2

    return outputImage.astype(float)

def main():
    """
    """

    pathRead = "/v/raid1a/gadluru/MRIdata/Cardiac/Verio/Astellas_2013/P120814/ReconData/mat_files/binned_recons"

    fileNameList = sorted(glob.glob("%s/*Interp*.mat" % pathRead))
    print(fileNameList)
    
    for ii in range(5):
        if ii == 0:
            data = matreader.MatReader(fileNameList[ii])
            data = data[np.newaxis,:,:,:]
            print(data.shape)
        elif ii < 5:
            data = np.append(data,
                             matreader.MatReader(fileNameList[ii])[np.newaxis,:,:,:],
                             axis=0)
        else:
            break

    print(data.shape)
    
    data = data.transpose(2,3,0,1)
    data = data[:,::-1,:,:]
    print(data.shape)
    # data = np.load(fileNameList[10])
    # print(fileNameList[3])
    
    data /= np.amax(data)


    X, y = GenerateHeartData.DataGenerator(data)

    sliceNumber = 42
    
    im1 = X[sliceNumber,:,:,0].squeeze()
    im2 = X[sliceNumber,:,:,1].squeeze()
    yTrue = y[sliceNumber,:,:,0].astype(float)
    
    sliceNumber = 100
    im1 = data[:,:,1,sliceNumber]
    im2 = data[:,:,3,sliceNumber]
    yTrue = data[:,:,2,sliceNumber]

    ySepConvFt = SepConvInterpolate.SepConv(im1,im2,"ft")
    yLinear = Linear(im1,im2)

    
    # plt.figure()
    # plt.imshow(ySepConvFt,cmap="gray")

    # plt.figure()
    # plt.imshow(yLinear,cmap="gray")
    
    # plt.figure()
    # plt.imshow(yTrue,cmap="gray")

    montageTrue = np.concatenate((im1,yTrue,im2),axis=1)    
    montageSepConv = np.concatenate((im1,ySepConvFt,im2),axis=1)
    montageLinear = np.concatenate((im1,yLinear,im2),axis=1)

    montage = np.concatenate((montageTrue,
                              montageSepConv,
                              montageLinear),axis=0)

    plt.figure()
    plt.imshow(im1,cmap="gray")
    plt.axis("off")
    plt.savefig("/home/mirl/egibbons/sepconv/images/im1.pdf",
            bbox_inches="tight",pad_inches=0)

    plt.figure()
    plt.imshow(ySepConvFt,cmap="gray")
    plt.axis("off")
    plt.savefig("/home/mirl/egibbons/sepconv/images/yout.pdf",
            bbox_inches="tight",pad_inches=0)

    plt.figure()
    plt.imshow(im2,cmap="gray")
    plt.axis("off")
    plt.savefig("/home/mirl/egibbons/sepconv/images/im2.pdf",
            bbox_inches="tight",pad_inches=0)


    plt.figure()
    plt.imshow(montage,cmap="gray",interpolation="hanning")
    plt.axis("off")
    plt.savefig("/home/mirl/egibbons/sepconv/images/montage_compare.pdf",
            bbox_inches="tight",pad_inches=0)


    plt.show()
    
if __name__ == "__main__":
    main()

