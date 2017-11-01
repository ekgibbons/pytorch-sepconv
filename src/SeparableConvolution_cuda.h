int SeparableConvolution_cuda_forward(THCudaTensor* input,
				      THCudaTensor* vertical,
				      THCudaTensor* horizontal,
				      THCudaTensor* output);

int SeparableConvolution_cuda_backward(THCudaTensor* gradLoss,
				       THCudaTensor* input,
				       THCudaTensor* vertical,
				       THCudaTensor* horizontal,
				       THCudaTensor* gradInput,				       
				       THCudaTensor* gradVertical,
				       THCudaTensor* gradHorizontal);
