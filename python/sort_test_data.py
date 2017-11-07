""" This parses the 30 or so cardiac cases into usable training data for the 
deep-slice project.  Need to get it running on python3.  At the moment I am 
just using python2.7 and it seems to working fine.  I am just lazy.
"""

from __future__ import print_function

import glob
import os
import random

import numpy as np
from scipy import io
from matplotlib import pyplot as plt

from matlab import matreader
from recon import imtools

def GenerateName(path):
    """
    """

    folders = []

    while 1:
        path, folder = os.path.split(path)
        
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)

            break

    name = "%s_%s" % (folders[4], folders[0])
    name = name[:-15]

    return name
    

def GetSliceNumber(fileName):
    """
    Returns the slice number based on location
    """

    sliceNumber = fileName.split("slice",1)[1][0]

    if sliceNumber in "_":
            sliceNumber = fileName.split("slice",1)[1][1]

    return int(sliceNumber)

def SortSlices(basePath):
    """
    """


    folderList = sorted(glob.glob("%s/P*" % basePath))
    folderList = folderList[-2:]

    numFiles = 0
    caseList = []

    for folder in folderList:
        if os.path.exists("%s/ReconData/mat_files/binned_recons/" %
                             folder): 
            
            reconDataFolder = ("%s/ReconData/mat_files/binned_recons/" %
                               folder)
        else:
            reconDataFolder = ("%s/ReconData/mat_files/" %
                               folder)
            
        print(reconDataFolder)
        matFileList = sorted(glob.glob("%s/*slice*.mat" % reconDataFolder))
        print(matFileList)

        if len(matFileList) == 0:
            print("empty")

        else:
            matFileListNew = [fileName for fileName in matFileList if "coil" not in fileName]
            
            ii = 1
            sublist = []
            for fileName in matFileListNew:
                sliceNumber = GetSliceNumber(fileName)
        
                if sliceNumber == ii:
                    sublist.append(fileName)

                ii +=1

                if ii == 6:
                    if len(sublist) == 5:
                        caseList.append(sublist)

                    sublist = []
                    ii = 1

    return caseList

def main():
    """
    """

    basePath = "/v/raid1a/gadluru/MRIdata/Cardiac/Verio/Astellas_2013/"
    
    sliceList = SortSlices(basePath)

    data4DAll = None
    firstCase = True

    numberCases = 0
    for sublist in sliceList:

        firstSlice = True
        data4D = None;
        
        for fileName in sublist:
            data = matreader.MatReader(fileName)
            print(data.shape)
            if len(data.shape) != 3:
                continue
            
            data = data.transpose(1,2,0)
     
            if np.prod(data.shape[:-1]) < 144*144:
                data = imtools.interpolate(data,144,144)
                
            if np.prod(data.shape[:-1]) > 144*144:
                data = data.transpose(0,2,1)
            
            fileNameUse = GenerateName(fileName)
            data4DSlice = abs(data[:,:,np.newaxis,:]).astype(np.float32)

            if firstSlice:
                data4D = data4DSlice
                firstSlice = False
            else:
                try:
                    data4D = np.append(data4D,data4DSlice,axis=2)
                except:
                    timePoints = data4D.shape[3]
                    timePointsSlice = data4DSlice.shape[3]

                    if timePoints < timePointsSlice:
                        data4DSlice = data4DSlice[:,:,:,:timePoints]
                    else: 
                        data4D = data4D[:,:,:,:timePointsSlice]

                    data4D = np.append(data4D,data4DSlice,axis=2)

        if data4D is not None:
            for ii in range(data4D.shape[3]):
                if np.sum(data4D[:,:,:,ii])  == 0.0:
                    data4D = data4D[:,:,:,:ii]
                    break

                data4D[:,:,:,ii] /= np.amax(data4D[:,:,:,ii])

        print(fileNameUse)
        print(data4D.shape)

        pathSave = "/v/raid1a/egibbons/data/deep-slice"

        np.save("%s/%s.npy" % (pathSave, fileNameUse),data4D)

    return 0


if __name__ == "__main__":
    main()
