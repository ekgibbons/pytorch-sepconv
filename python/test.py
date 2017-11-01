import torch
import time

from SeparableConvolution import SeparableConvolution

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
    # end

    def forward(self, inputImage, vertical, horizontal):
        return SeparableConvolution()(inputImage, vertical, horizontal)
    # end
# end

# if True:
#     moduleTest = Network().cuda()

#     variableInput = torch.autograd.Variable(data=torch.randn(2, 1, 9, 9).cuda(), requires_grad=False) # NO GRADIENT
#     variableVertical = torch.autograd.Variable(data=torch.randn(2, 9, 1, 1).cuda(), requires_grad=True)
#     variableHorizontal = torch.autograd.Variable(data=torch.randn(2, 9, 1, 1).cuda(), requires_grad=False)

#     variableParameters = tuple([ variableInput, variableVertical, variableHorizontal ])

#     print("start grad check")
#     print("gradcheck", torch.autograd.gradcheck(func=moduleTest, inputs=variableParameters, eps=0.001, atol=0.001), "should be true")
# end


net = Network().cuda()
for i in range(1):
    inputImage = torch.rand(1, 1, 9, 9).cuda()
    vertical = torch.rand(1,51,1,1).cuda()
    horizontal = torch.rand(1,51,1,1).cuda()
    
    inputImageVar = torch.autograd.Variable(inputImage, requires_grad=True)
    verticalVar = torch.autograd.Variable(vertical, requires_grad=True)
    horizontalVar = torch.autograd.Variable(horizontal, requires_grad=True)

    output = net(inputImageVar, verticalVar, horizontalVar)
    
    inputImage = inputImage.squeeze_()
    horizontal = horizontal.squeeze_().unsqueeze(1)
    vertical = vertical.squeeze_().unsqueeze(0)

    stage1 = torch.mm(inputImage, horizontal)
    expected = torch.mm(vertical, stage1)

    
    print(output)
    print(expected)

    print(torch.sum(output.data - torch.autograd.Variable(expected).data),
          '<-- should be 0.0')

    output.backward(output.data)
# end

print('switching to DataParallel mode')

# net = torch.nn.DataParallel(Network()).cuda()
# for i in range(10):
# 	input1 = torch.rand(64, 128, 128).cuda()
# 	input2 = torch.rand(64, 128, 1).cuda()

# 	input1 = torch.autograd.Variable(input1, requires_grad=True)
# 	input2 = torch.autograd.Variable(input2, requires_grad=True)

# 	output = net(input1, input2)
# 	expected = torch.bmm(input1, input2)

# 	print(torch.sum(output.data - expected.data), '<-- should be 0.0')

# 	# output.backward(output.data)
# # end
