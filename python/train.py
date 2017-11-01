import time
import os
import copy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import numpy as np

import torch
import torch.utils.serialization
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

from matplotlib import pyplot as plt

import PIL
import PIL.Image
from skimage import color

import GenerateHeartData
import DataLoader
import model
import VggLoss
from SeparableConvolution import SeparableConvolution 
from recon import imtools

def TrainModel(dataLoaders, datasetSizes, model, 
               lossFunction, optimizer, scheduler, 
               numEpochs=25, vggModel=None):
    """
    """
    print("\n")
    print("Start Training")
    

    since = time.time()

    bestLoss = 1e15
    bestModelWeights = model.state_dict()

    for epoch in range(numEpochs):
        epochStart = time.time()
        print("Epoch {}/{}".format(epoch + 1, numEpochs))
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train(True)
            else:
                model.train(False)
            
            runningLoss = 0.0

            for data in dataLoaders[phase]:
                inputs, labels = data

                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                optimizer.zero_grad()

                # this is where the module gets called
                outputs = model(inputs)
                loss = lossFunction(outputs, labels)
                # print(loss.data[0])
                
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                runningLoss += loss.data[0]

            epochLoss = runningLoss/datasetSizes[phase]

            print("{} Loss: {:.4g}".format(phase,epochLoss))
                                                        
            
            if phase == "val" and epochLoss < bestLoss:
                print("val Loss decreased from: {:0.4g} to {:0.4g}...saving model".format(bestLoss,
                                                                                          epochLoss))
                bestLoss = epochLoss
                torch.save(model.state_dict(),"../models/network-ft.pytorch")
                bestModelWeights = copy.deepcopy(model.state_dict())

        timeEpoch = time.time() - epochStart
        print("time for epoch: {:.0f}m {:.0f}s".format(timeEpoch // 60, 
                                                       timeEpoch % 60))


    timeElapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(timeElapsed // 60, 
                                                        timeElapsed % 60))
    print("Best val loss: {:4f}".format(bestLoss))

    model.load_state_dict(bestModelWeights)
    
    return model


def main():
    """
    """
    
    torch.cuda.device(0) # change this if you have a multiple graphics cards and you want to utilize them
    
    use_gpu = torch.cuda.is_available()
    
    # make sure to use cudnn for computational performance
    torch.backends.cudnn.enabled = True 

    # The network
    moduleNetwork = model.Network().cuda()
    moduleNetwork.load_state_dict(torch.load('../models/network-lf.pytorch'))
    
    # Hyperparameters
    beta1 = 0.9
    beta2 = 0.999
    learningRate = 0.001
    batchSize = 16

    # Determine the loss function
    # lossFunction = torch.nn.L1Loss()
    lossFunction = VggLoss.VggLoss()
    lossFunction.cuda()
    
    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer = optim.Adamax(moduleNetwork.parameters(), 
                             lr=learningRate, 
                             betas=(beta1,beta2),
                             eps=1e-8)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    lrScheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print("Load data...")
    pathRead = "/v/raid1a/egibbons/data/deep-slice"

    dataRunning = None
    for ii in range(8):  # after 8 the data gets sketchy...
        dataTemp = np.load("%s/training_hearts_%02i.npy" % (pathRead, ii))
        if (ii != 0):
            dataRunning = np.concatenate((dataRunning,dataTemp),axis=3)
        elif ii is 0:
            dataRunning = dataTemp

    print(dataRunning.shape)
    dataRunning = imtools.crop(dataRunning,128,128)
    print(dataRunning.shape)

    # set up the data
    X, y = GenerateHeartData.DataGenerator(dataRunning)

    imEx = np.concatenate((X[1000,:,:,0].squeeze(),
                           y[1000,:,:,0].squeeze(),
                           X[1000,:,:,1].squeeze()),axis=1)

    # plt.figure()
    # plt.imshow(imEx,cmap="gray")
    # plt.show()

    XRgb = np.zeros((X.shape[0],3*2,X.shape[1],X.shape[2]))
    yRgb = np.zeros((y.shape[0],3,y.shape[1],y.shape[2]))
    
    for ii in range(X.shape[0]):
        im0 = color.grey2rgb(X[ii,:,:,0].squeeze())
        im1 = color.grey2rgb(X[ii,:,:,1].squeeze())
        im2 = color.grey2rgb(y[ii,:,:,0].squeeze())

        XRgb[ii,:3,:,:] = np.rollaxis(im0,2,0)[np.newaxis,:,:,:]
        XRgb[ii,3:,:,:] = np.rollaxis(im1,2,0)[np.newaxis,:,:,:]
        yRgb[ii,:,:,:] = np.rollaxis(im2,2,0)[np.newaxis,:,:,:]

    XTensor = torch.FloatTensor(XRgb)
    yTensor = torch.FloatTensor(yRgb)

    print(XTensor.size())
    print(yTensor.size())

    # set up the data loaders
    numTrain = XTensor.size(0)
    validSize = 0.2


    trainLoader, valLoader = DataLoader.GetTrainValLoader(XTensor,yTensor,
                                                          batch_size=batchSize,
                                                          valid_size=validSize)

    dataLoaders = {}
    dataLoaders["train"] = trainLoader
    dataLoaders["val"]  = valLoader

    datasetSizes = {}
    datasetSizes["train"] = numTrain - int(np.floor(validSize*numTrain))
    datasetSizes["val"] = int(np.floor(validSize*numTrain))
    print(datasetSizes)

    # Do the training
    TrainModel(dataLoaders, datasetSizes, moduleNetwork, 
               lossFunction, optimizer,
               lrScheduler, numEpochs=200)
    


if __name__ == "__main__":
    main()
