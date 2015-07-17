all: lss lss_generate

lss: lss.cu
	nvcc -arch=sm_35 -rdc=true lss.cu ../../include/log_helper.cu -I../../include/ -o lss -lcudadevrt

lss_generate: lss_generate.cu
	nvcc -arch=sm_35 -rdc=true lss_generate.cu -o lss_generate -lcudadevrt

clean:
	rm -rf lss lss_generate
