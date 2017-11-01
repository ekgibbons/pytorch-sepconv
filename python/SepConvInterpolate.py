import math
import numpy as np
import torch
import torch.utils.serialization
from skimage import color

import model
from SeparableConvolution import SeparableConvolution 

def SepConv(im1,im2,modelType="ft"):
    """
    """

    torch.cuda.device(1)     
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    moduleNetwork = model.Network().cuda()
    moduleNetwork.load_state_dict(torch.load("../models/network-%s.pytorch" % modelType))

    # get dimensionality of image correct
    im1 = np.rollaxis(color.gray2rgb(im1),2,0)
    im2 = np.rollaxis(color.gray2rgb(im2),2,0)

    tensorInputFirst = torch.FloatTensor(im1)
    tensorInputSecond = torch.FloatTensor(im2)
    tensorOutput = torch.FloatTensor()
    
    assert(tensorInputFirst.size(1) == tensorInputSecond.size(1))
    assert(tensorInputFirst.size(2) == tensorInputSecond.size(2))
    
    intWidth = tensorInputFirst.size(2)
    intHeight = tensorInputFirst.size(1)
    
    assert(intWidth <= 1280)
    assert(intHeight <= 720)
    
    intPaddingLeft = int(math.floor(51 / 2.0))
    intPaddingTop = int(math.floor(51 / 2.0))
    intPaddingRight = int(math.floor(51 / 2.0))
    intPaddingBottom = int(math.floor(51 / 2.0))
    modulePaddingInput = torch.nn.Module()
    modulePaddingOutput = torch.nn.Module()

    if True:
        intPaddingWidth = intPaddingLeft + intWidth + intPaddingRight
        intPaddingHeight = intPaddingTop + intHeight + intPaddingBottom
        
        if intPaddingWidth != ((intPaddingWidth >> 7) << 7):
            intPaddingWidth = (((intPaddingWidth >> 7) + 1) << 7) # more than necessary
        # end

        if intPaddingHeight != ((intPaddingHeight >> 7) << 7):
            intPaddingHeight = (((intPaddingHeight >> 7) + 1) << 7) # more than necessary
        # end

        intPaddingWidth = intPaddingWidth - (intPaddingLeft + intWidth + intPaddingRight)
        intPaddingHeight = intPaddingHeight - (intPaddingTop + intHeight + intPaddingBottom)
        
        modulePaddingInput = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight + intPaddingWidth, 
                                                        intPaddingTop, intPaddingBottom + intPaddingHeight])
        
        modulePaddingOutput = torch.nn.ReplicationPad2d([0 - intPaddingLeft, 0 - intPaddingRight - intPaddingWidth, 
                                                         0 - intPaddingTop, 0 - intPaddingBottom - intPaddingHeight])
    # end

    if True:
        tensorInputFirst = tensorInputFirst.cuda()
        tensorInputSecond = tensorInputSecond.cuda()
        tensorOutput = tensorOutput.cuda()
    
        modulePaddingInput = modulePaddingInput.cuda()
        modulePaddingOutput = modulePaddingOutput.cuda()
    # end

    if True:
        variablePaddingFirst = modulePaddingInput(torch.autograd.Variable(data=tensorInputFirst.view(1, 3, 
                                                                                                     intHeight, intWidth),
                                                                          volatile=True))
        variablePaddingSecond = modulePaddingInput(torch.autograd.Variable(data=tensorInputSecond.view(1, 3, 
                                                                                                       intHeight, intWidth),
                                                                           volatile=True))

        variableInputCat = torch.cat((variablePaddingFirst,variablePaddingSecond),dim=1)
        variablePaddingOutput = modulePaddingOutput(moduleNetwork(variableInputCat))

        tensorOutput.resize_(3, intHeight, intWidth).copy_(variablePaddingOutput.data[0])
        
    # end

    if True:
        tensorInputFirst = tensorInputFirst.cpu()
        tensorInputSecond = tensorInputSecond.cpu()
        tensorOutput = tensorOutput.cpu()
    # end

    outputImage = tensorOutput.clamp(0.0,1.0).numpy()
    outputImage = np.rollaxis(tensorOutput.numpy(), 0, 3)

    outputImageGray = color.rgb2gray(outputImage).astype(float)
 
    return outputImageGray

