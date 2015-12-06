#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <math.h>
#include <cuda.h>

int main (int arg, char* argv[]) {
    
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,device);

    printf("Multi Processor Count: %d", prop.multiProcessorCount);
}
