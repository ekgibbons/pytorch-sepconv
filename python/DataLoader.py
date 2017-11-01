# This is an example for the CIFAR-10 dataset.
# There's a function for creating a train and validation iterator.
# There's also a function for creating a test iterator.
# Inspired by https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
# Website:  https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb


import multiprocessing
import numpy as np
import torch
from torch.utils import data
from skimage import color

import GenerateHeartData

def GetTrainValLoader(X, y,
                      batch_size=50,
                      random_seed=5000,
                      valid_size=0.2,
                      shuffle=True,
                      num_workers=multiprocessing.cpu_count(),
                      pin_memory=False):
    """
    Utility function for loading and returning train and valid 
    multi-process iterators over the CIFAR-10 dataset. A sample 
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg


    num_train = int(X.size(0))
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = data.sampler.SubsetRandomSampler(valid_idx)

    train_dataset = data.TensorDataset(X,y) 
    valid_dataset = data.TensorDataset(X,y) 

    train_loader = data.DataLoader(train_dataset, 
                                   batch_size=batch_size, 
                                   sampler=train_sampler, 
                                   num_workers=num_workers, 
                                   pin_memory=pin_memory)
    
    valid_loader = data.DataLoader(valid_dataset, 
                                   batch_size=batch_size, 
                                   sampler=valid_sampler, 
                                   num_workers=num_workers, 
                                   pin_memory=pin_memory)

    return (train_loader, valid_loader)
    
def GetTestLoader(data_dir, 
                  batch_size,
                  shuffle=True,
                  num_workers=4,
                  pin_memory=False):
    """
    Utility function for loading and returning a multi-process 
    test iterator over the CIFAR-10 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """

    dataset = data.TensorDataset(XTensor,yTensor) 

    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    return data_loader

def main():
    """
    """
    
    print("Load data...")
    pathRead = "/v/raid1a/egibbons/data/deep-slice"
    
    dataRunning = None
    for ii in range(1):
        dataTemp = np.load("%s/training_hearts_%02i.npy" % (pathRead, ii))
        if (ii != 0) and (ii != 9):
            dataRunning = np.concatenate((dataRunning,dataTemp),axis=3)
        elif ii is 0:
            dataRunning = dataTemp
    
    # set up the data
    X, y = GenerateHeartData.DataGenerator(dataRunning)
    
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
    
    trainLoader, valLoader = GetTrainValLoader(XTensor,yTensor,100)
    
if __name__ == "__main__":
    main()


    # # define transforms
    # valid_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize
    #     ])
    # if augment:
    #     train_transform = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize
    #     ])
    # else:
    #     train_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize
    #     ])

    # # load the dataset
    # train_dataset = datasets.CIFAR10(root=data_dir, train=True, 
    #                                  download=True, transform=train_transform)

    # valid_dataset = datasets.CIFAR10(root=data_dir, train=True, 
    #                                  download=True, transform=valid_transform)
