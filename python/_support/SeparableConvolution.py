import time
import torch

import _ext.cunnex

FILT_LENGTH =  51

class SeparableConvolution(torch.autograd.Function):
    def __init__(self):
        super(SeparableConvolution, self).__init__()
    # end

    def forward(self, input, vertical, horizontal):
        self.save_for_backward(input, vertical, horizontal)

        intBatches = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert(intInputHeight - FILT_LENGTH == intOutputHeight - 1)
        assert(intInputWidth - FILT_LENGTH == intOutputWidth - 1)
        assert(intFilterSize == FILT_LENGTH)
        
        assert(input.is_contiguous() == True)
        assert(vertical.is_contiguous() == True)
        assert(horizontal.is_contiguous() == True)

        output = input.new().resize_(intBatches, 
                                     intInputDepth, 
                                     intOutputHeight, 
                                     intOutputWidth).zero_()
        
        if input.is_cuda == True:
            _ext.cunnex.SeparableConvolution_cuda_forward(
                input,
                vertical,
                horizontal,
                output
            )
        # end

        return output
    # end
    
    def backward(self, gradOutput):

        input, vertical, horizontal = self.saved_tensors

        gradInput = input.new().resize_(input.size()).zero_()
        gradVertical = vertical.new().resize_(vertical.size()).zero_()
        gradHorizontal = horizontal.new().resize_(horizontal.size()).zero_()


        if gradOutput.is_cuda == True:
            _ext.cunnex.SeparableConvolution_cuda_backward(
                gradOutput,
                input,
                vertical,
                horizontal,
                gradInput,
                gradVertical,
                gradHorizontal
            )
        # end
         
        return gradInput, gradVertical, gradHorizontal
    #end
#end

def main():
    """
    """
    print("Starting gradient tests...")
    
    from torch.autograd import Variable, gradcheck
    inputs = (Variable(torch.randn(2,3,FILT_LENGTH,FILT_LENGTH).cuda(), requires_grad=False),
              Variable(torch.randn(2,FILT_LENGTH,1,1).cuda(), requires_grad=True),
              Variable(torch.randn(2,FILT_LENGTH,1,1).cuda(), requires_grad=True),
          )
    test = gradcheck(SeparableConvolution(),inputs,eps=1e-3,atol=1e-3,rtol=1e-3)
                     # eps=1e-3,atol=1e-)
    
    print(test)

# end        

    
if __name__ == "__main__":
    main()

# end
