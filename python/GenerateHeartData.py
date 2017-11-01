from __future__ import print_function

import glob
import os

import numpy as np
from scipy import io
from matplotlib import pyplot as plt

        

def DataGenerator(data):
    """
    """
    
    ny, nx, ns, nc = data.shape

    firstFrameBatch = np.zeros(shape=(3*nc,ny,nx),dtype=np.float32)
    middleFrameBatch = np.zeros(shape=(3*nc,ny,nx),dtype=np.float32)
    lastFrameBatch = np.zeros(shape=(3*nc,ny,nx),dtype=np.float32)

    for ii in range(nc):
        
        firstFrameBatch[ii] = data[:,:,0,ii]
        middleFrameBatch[ii] = data[:,:,1,ii]
        lastFrameBatch[ii] = data[:,:,2,ii]

        firstFrameBatch[ii+nc//3] = data[:,:,1,ii]
        middleFrameBatch[ii+nc//3] = data[:,:,2,ii]
        lastFrameBatch[ii+nc//3] = data[:,:,3,ii]

        firstFrameBatch[ii+2*nc//3] = data[:,:,2,ii]
        middleFrameBatch[ii+2*nc//3] = data[:,:,3,ii]
        lastFrameBatch[ii+2*nc//3] = data[:,:,4,ii]

    firstFrameBatch = firstFrameBatch[:,:,:,np.newaxis]
    middleFrameBatch = middleFrameBatch[:,:,:,np.newaxis]
    lastFrameBatch = lastFrameBatch[:,:,:,np.newaxis]

    y = middleFrameBatch
    X = np.concatenate((firstFrameBatch,lastFrameBatch),axis=3)

    return X, y

def main():
    """
    """

    pathRead = "/v/raid1a/egibbons/data/deep-slice"
    data = np.load("%s/training_hearts_00.npy" % pathRead)

    print(data.shape)

    X, y = DataGenerator(data)

    XTemp0 = X[50,:,:,0].squeeze()
    XTemp1 = X[50,:,:,1].squeeze()
    yTemp0 = y[50,:,:,0].squeeze()

    im = np.concatenate((XTemp0,yTemp0,XTemp1),axis=1)

    plt.figure()
    plt.imshow(abs(im))
    plt.show()



if __name__ == "__main__":
    main()
