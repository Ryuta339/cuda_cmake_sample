#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"

__global__ void kernel_vecDouble(int *in, int *out, const size_t n) {
	int i = threadIdx.x;
	if (i < n) {
		out[i] = in[i] * 2;
	}
}

template<class T>
class Doubler {
	T obj;
	public:
		void vecDouble(int *in, int *out, const size_t n) {
			obj.vecDouble(in, out, n);
		}
};

class HostDoubler {
	public:
		void vecDouble(int *in, int *out, const size_t n) {
			for (size_t i=0; i<n; i++)
				out[i] = in[i] * 2;
		}
};

class DeviceDoubler {
	public:
		static void vecDouble(int *hIn, int *hOut, const size_t n) {
			int *dIn;
			int *dOut;
			cudaMalloc((void **)&dIn, n * sizeof(int));
			cudaMalloc((void **)&dOut, n * sizeof(int));
			cudaMemcpy(dIn, hIn, n * sizeof(int), cudaMemcpyHostToDevice);

			kernel_vecDouble<<<1, n>>>(dIn, dOut, n);
			cudaDeviceSynchronize();

			cudaMemcpy(hOut, dOut, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dIn);
			cudaFree(dOut);
		}
};

static void vecDouble(int *, int *, const size_t);

int main() {
	printf("Hello\n");

	const size_t n = 10;
	int * in = new int[n];
	int * out = new int[n];
	int * answer = new int[n];

	for (size_t i=0; i<n; i++)
		in[i] = rand() % 100;

	Doubler<HostDoubler> host;
	Doubler<DeviceDoubler> device;

	host.vecDouble(in, answer, n);
	device.vecDouble(in, out, n);

	for (size_t i=0; i<n; i++) {
		if (answer[i] != out[i]) {
			printf("error at index %d\n", i);
			break;
		}
	}
	printf("OK\n");

	delete[] in;
	delete[] out;
	delete[] answer;

	return 0;
}
