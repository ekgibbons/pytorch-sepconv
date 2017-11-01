TORCH=/home/mirl/egibbons/.conda/envs/tf2/lib/python3.6/site-packages/torch

main: 
	/usr/local/cuda/bin/nvcc -c -o src/SeparableConvolution_kernel.o src/SeparableConvolution_kernel.cu --gpu-architecture=compute_52 --gpu-code=compute_52 --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC
	python install.py
	rm -rf python/_ext
	mv -f _ext python/
