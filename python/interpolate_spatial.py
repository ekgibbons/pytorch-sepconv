import glob
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"       

import numpy as np
from matplotlib import pyplot as plt

import GenerateHeartData
import SepConvInterpolate
from matlab import matreader

def NaiveInterpolation(im1,im2):
    """
    """

    output = (im1 + im2)/2

    return output


# basePath = "/v/raid1a/egibbons/data/deep-slice"

# listFiles =  sorted(glob.glob("%s/*MID*.npy" % basePath))

# npyFile = listFiles[2]
# print("These are test files")
# print(listFiles)
# print("This is what I chose...")
# print(npyFile)

# data = np.load(npyFile)

# pathData = "/v/raid1a/ytian/MRIdata/Cardiac/Prisma/P103117/processing"
# listFiles = sorted(glob.glob("%s/*.mat" % pathData))
# fileUse = listFiles[4]

pathData = "/v/raid1a/ytian/MRIdata/Cardiac/Prisma/P092917/processing"
fileUse = "%s/SET09476_MID160_GROG_1_1x1_Rest_171005_001554.mat" % pathData
fileUse = "%s/SET09473_MID121_GROG_1_1x1_Stress_171005_000239.mat" % pathData


print(fileUse)
data = matreader.MatReader(fileUse,"Image")
print(data.shape)

data = data.transpose(4,3,2,0,1)

# print(data.shape)

# for ii in range(data.shape[2]):
#     plt.figure(1)
#     plt.imshow(abs(data[:,:,ii,0,0]))
#     plt.pause(0.02)

# plt.show()

timeFrame = 42
montage = np.concatenate((data[:,:,timeFrame,0,2],
                          data[:,:,timeFrame,1,1],
                          data[:,:,timeFrame,1,2],
                          data[:,:,timeFrame,2,1],
                          data[:,:,timeFrame,2,2]),axis=1)

plt.figure()
plt.imshow(montage,cmap="gray")

for ii in range(3):
    for jj in range(3):
        data[:,:,:,ii,jj] /= np.amax(data[:,:,:,ii,jj])

data /= np.amax(data)

timeFrame = 42
montage = np.concatenate((data[:,:,timeFrame,0,2],
                          data[:,:,timeFrame,1,1],
                          data[:,:,timeFrame,1,2],
                          data[:,:,timeFrame,2,1],
                          data[:,:,timeFrame,2,2]),axis=1)

plt.figure()
plt.imshow(montage,cmap="gray")
plt.show()

interpolated = np.zeros_like(data)
naive = np.zeros_like(data)

for ii in range(interpolated.shape[2]):
    interpolated[:,:,ii,1,2] = SepConvInterpolate.SepConv(data[:,:,ii,0,2],
                                                          data[:,:,ii,1,2],
                                                          "ft")
    
    interpolated[:,:,ii,2,2] = SepConvInterpolate.SepConv(data[:,:,ii,1,2],
                                                          data[:,:,ii,2,2],
                                                          "ft")

    naive[:,:,ii,1,2] = NaiveInterpolation(data[:,:,ii,0,2],
                                           data[:,:,ii,1,2])
    
    naive[:,:,ii,2,2] = NaiveInterpolation(data[:,:,ii,1,2],
                                           data[:,:,ii,2,2])
                                                        

montageInterpolate = np.concatenate((data[:,:,timeFrame,0,2],
                                     interpolated[:,:,timeFrame,1,2],
                                     data[:,:,timeFrame,1,2],
                                     interpolated[:,:,timeFrame,2,2],
                                     data[:,:,timeFrame,2,2]),axis=1)


montageNaive = np.concatenate((data[:,:,timeFrame,0,2],
                               naive[:,:,timeFrame,1,2],
                               data[:,:,timeFrame,1,2],
                               naive[:,:,timeFrame,2,2],
                               data[:,:,timeFrame,2,2]),axis=1)



montageCombined = np.concatenate((montage,montageInterpolate,montageNaive),
                                 axis=0)

plt.figure()
plt.imshow(montageCombined,cmap="gray")
plt.axis("off")
plt.savefig("/home/mirl/egibbons/sepconv/images/montage_slices.pdf",
            bbox_inches="tight",pad_inches=0)


for ii in range(10):
    if ii is 0:
        montageInterpolate = interpolated[:,:,ii+27,1,2]
        montageTrue = data[:,:,ii+27,1,1]
        montageNaive = naive[:,:,ii+27,1,2]

    else:
        montageInterpolate = np.append(montageInterpolate,
                                       interpolated[:,:,ii+27,1,2],
                                       axis=1)
        
        montageTrue = np.append(montageTrue,
                                data[:,:,ii+27,1,1],
                                axis=1)
        
        montageNaive = np.append(montageNaive,
                                 naive[:,:,ii+27,1,2],
                                 axis=1)

        
montageCombine = np.concatenate((montageTrue, 
                                 montageInterpolate, 
                                 montageNaive), 
                                axis=0)

plt.figure()
plt.imshow(montageCombine,cmap="gray")
plt.axis("off")
plt.savefig("/home/mirl/egibbons/sepconv/images/montage_temporal.pdf",
            bbox_inches="tight",pad_inches=0)




plt.show()


