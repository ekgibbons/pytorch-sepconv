import glob
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"       

import numpy as np
from matplotlib import pyplot as plt

import GenerateHeartData
import SepConvInterpolate
from matlab import matreader
# basePath = "/v/raid1a/egibbons/data/deep-slice"

# listFiles =  sorted(glob.glob("%s/*MID*.npy" % basePath))

# npyFile = listFiles[2]
# print("These are test files")
# print(listFiles)
# print("This is what I chose...")
# print(npyFile)

# data = np.load(npyFile)

# pathData = "/v/raid1a/ytian/MRIdata/Cardiac/Prisma/P103117/processing"
# pathData = "/v/raid1a/ytian/MRIdata/Cardiac/Prisma/P092917/processing"
# listFiles = sorted(glob.glob("%s/*.mat" % pathData))

# fileUse = listFiles[4]

pathData = "/v/raid1a/ytian/MRIdata/Cardiac/Prisma/P092917/processing"
fileUse = "%s/SET09476_MID160_GROG_1_1x1_Rest_171005_001554.mat" % pathData
fileUse = "%s/SET09473_MID121_GROG_1_1x1_Stress_171005_000239.mat" % pathData

print(fileUse)
data = matreader.MatReader(fileUse,keyName="Image")
print(data.shape)

data = data.transpose(4,3,2,0,1)

# print(data.shape)

# for ii in range(data.shape[2]):
#     plt.figure(1)
#     plt.imshow(abs(data[:,:,ii,0,0]))
#     plt.pause(0.02)

# plt.show()


maxValue = np.amax(data)
data = data/maxValue

# montage = np.concatenate((data[:,:,1,100],
#                           data[:,:,2,100],
#                           data[:,:,3,100]),axis=1)

# plt.figure()
# plt.imshow(montage,cmap="gray")

temporalTrue = data[:,:,:,1,1].squeeze()
temporalInterpolate = np.zeros_like(temporalTrue)

print(temporalTrue.shape)

for ii in range(temporalTrue.shape[2]):
    if ii % 2:
        if (ii + 1) == temporalTrue.shape[2]:
            break
            
        temporalInterpolate[:,:,ii] = SepConvInterpolate.SepConv(temporalTrue[:,:,ii-1],
                                                                 temporalTrue[:,:,ii+1],
                                                                 "ft-old")
    else:
        temporalInterpolate[:,:,ii] = temporalTrue[:,:,ii]

montage = np.concatenate((temporalTrue,temporalInterpolate),axis=1)

for ii in range(temporalTrue.shape[2]):
    plt.figure(1)
    plt.imshow(abs(montage[:,:,ii]))
    plt.title("Frame number: %i" % ii)
    plt.pause(0.02)

plt.show()


for ii in range(20):
    if ii is 0:
        montageInterpolate = temporalInterpolate[:,:,ii+20]
        montageTrue = temporalTrue[:,:,ii+20]
    else:
        montageInterpolate = np.append(montageInterpolate,
                                       temporalInterpolate[:,:,ii+20],
                                       axis=1)

        montageTrue = np.append(montageTrue,
                                temporalTrue[:,:,ii+20],
                                axis=1)

montageCombine = np.concatenate((montageTrue, montageInterpolate), 
                                axis=0)

plt.figure()
plt.imshow(montageCombine)

plt.show()
