import glob
import os
import warnings

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
    fileNameList = sorted(glob.glob("%s/*Pre*.npy" % pathRead))
    print(fileNameList)

    data = None

    for fileName in fileNameList:
        dataTemp = np.load(fileName)

        if data is None:
            data = dataTemp
        else:
            data = np.append(data,dataTemp,axis=3)

    data /= np.amax(data)
    print(np.sum(np.isinf(data)))

    # data = np.load("%s/training_hearts_00.npy" % (pathRead))
    print(data.shape)


    X, y = GenerateHeartData.DataGenerator(data)
    print(X.shape)    

    sliceExtent = X.shape[0]
    # sliceExtent = 1000


    
    indexList = np.arange(0,X.shape[0])
    print(indexList)
    np.random.shuffle(indexList)
    print(indexList)
    indexList = indexList[:sliceExtent]
    
    X = X[indexList,:,:,:]
    y = y[indexList,:,:,:]


    ssimSepConv = np.zeros(sliceExtent)
    ssimLinear = np.zeros(sliceExtent)

    psnrSepConv = np.zeros(sliceExtent)
    psnrLinear = np.zeros(sliceExtent)

    mseSepConv = np.zeros(sliceExtent)
    mseLinear = np.zeros(sliceExtent)

    ii = 0
    for sliceNumber in range(sliceExtent):
    
        ii += 1
        
        print("slice: %i/%i" % (sliceNumber,sliceExtent))

        im1 = X[sliceNumber,:,:,0].squeeze()
        im2 = X[sliceNumber,:,:,1].squeeze()

        warnings.simplefilter("error", RuntimeWarning)

        try:
            ySepConv = SepConvInterpolate.SepConv(im1,im2,"ft")
            yLinear = Linear(im1,im2)
            yTrue = y[sliceNumber,:,:,0].astype(float)

            ssimSepConv[sliceNumber] = measure.compare_ssim(ySepConv,yTrue)
            ssimLinear[sliceNumber] = measure.compare_ssim(yLinear,yTrue)
            
            psnrSepConv[sliceNumber] = measure.compare_psnr(ySepConv,yTrue)
            psnrLinear[sliceNumber] = measure.compare_psnr(yLinear,yTrue)

            mseSepConv[sliceNumber] = measure.compare_mse(ySepConv,yTrue)
            mseLinear[sliceNumber] = measure.compare_mse(yLinear,yTrue)

                
        except RuntimeWarning:
            print("Ran into conversion error...skipping this case")
            continue
        except:
            continue
        


    print("SSIM")
    print("\tSepCon -- mean: %f std: %f" % (np.mean(ssimSepConv),np.std(ssimSepConv)))
    print("\tLinear -- mean: %f std: %f" % (np.mean(ssimLinear),np.std(ssimLinear)))
    print("\n")

    print("PSNR")
    print("\tSepCon -- mean: %f std: %f" % (np.mean(psnrSepConv),np.std(psnrSepConv)))
    print("\tLinear -- mean: %f std: %f" % (np.mean(psnrLinear),np.std(psnrLinear)))
    print("\n")

    print("NRMSE")
    print("\tSepCon -- mean: %f std: %f" % (np.mean(mseSepConv),np.std(mseSepConv)))
    print("\tLinear -- mean: %f std: %f" % (np.mean(mseLinear),np.std(mseLinear)))
    print("\n")


    plt.show()
    
if __name__ == "__main__":
    main()

