import math
import torch

from _support.SeparableConvolution import SeparableConvolution

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
            )
        # end

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVertical1 = Subnet()
        self.moduleVertical2 = Subnet()
        self.moduleHorizontal1 = Subnet()
        self.moduleHorizontal2 = Subnet()
        
        self.modulePad = torch.nn.ReplicationPad2d([ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ])

    # end

    def forward(self, variableCat): # variableInput1, variableInput2):
        
        variableInput1 = variableCat[:,:3,:,:]
        variableInput2 = variableCat[:,3:,:,:]

        variableJoin = torch.cat([variableInput1, variableInput2], 1)
        
        variableConv1 = self.moduleConv1(variableJoin)
        variablePool1 = self.modulePool1(variableConv1)
        
        variableConv2 = self.moduleConv2(variablePool1)
        variablePool2 = self.modulePool2(variableConv2)
        
        variableConv3 = self.moduleConv3(variablePool2)
        variablePool3 = self.modulePool3(variableConv3)
        
        variableConv4 = self.moduleConv4(variablePool3)
        variablePool4 = self.modulePool4(variableConv4)
        
        variableConv5 = self.moduleConv5(variablePool4)
        variablePool5 = self.modulePool5(variableConv5)
        
        variableDeconv5 = self.moduleDeconv5(variablePool5)
        variableUpsample5 = self.moduleUpsample5(variableDeconv5)

        variableDeconv4 = self.moduleDeconv4(variableUpsample5 + variableConv5)
        variableUpsample4 = self.moduleUpsample4(variableDeconv4)
        
        variableDeconv3 = self.moduleDeconv3(variableUpsample4 + variableConv4)
        variableUpsample3 = self.moduleUpsample3(variableDeconv3)
        
        variableDeconv2 = self.moduleDeconv2(variableUpsample3 + variableConv3)
        variableUpsample2 = self.moduleUpsample2(variableDeconv2)
        
        variableCombine = variableUpsample2 + variableConv2
        
        variableDot1 = SeparableConvolution()(self.modulePad(variableInput1), self.moduleVertical1(variableCombine), self.moduleHorizontal1(variableCombine))
        variableDot2 = SeparableConvolution()(self.modulePad(variableInput2), self.moduleVertical2(variableCombine), self.moduleHorizontal2(variableCombine))

        return variableDot1 + variableDot2
    # end
# end
