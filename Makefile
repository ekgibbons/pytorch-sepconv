TORCH=/home/mirl/egibbons/.conda/envs/tf2/lib/python3.6/site-packages/torch
INSTALL_DIR=/home/mirl/egibbons/python_utils/recon

main: 
	/usr/local/cuda/bin/nvcc -c -o src/SeparableConvolution_kernel.o src/SeparableConvolution_kernel.cu --gpu-architecture=compute_52 --gpu-code=compute_52 --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC
	python install.py
	rm -rf python/_support/_ext
	mv -f _ext python/_support/

build: FORCE
	rm -rf build/
	mkdir build
	mkdir build/_support
	cp python/__init__.py build/
	cp python/SepConvInterpolate.py build/
	cp -r python/_support/_ext build/_ext
	cp python/_support/SeparableConvolution.py build/_support
	cp python/_support/model.py build/_support
	cp python/_support/__init__.py build/_support

install: FORCE
	make build
	rm -rf ${INSTALL_DIR}/sepconv
	mv build/ ${INSTALL_DIR}/sepconv

FORCE: ;
