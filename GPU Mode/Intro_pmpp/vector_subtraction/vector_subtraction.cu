#include <cuda.h>
#include <stdio.h>

// compute vector subtract `C = A - B`
// each thread performs one pair-wise subtraction
__global__ void vecSubKernel(float *A, float *B, float *C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < n) {
    C[i] = A[i] - B[i];
  }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

// Wrapper
void vecSub(float *A, float *B, float *C, int n) {
  float *A_d, *B_d, *C_d;
  size_t size = n * sizeof(float);

  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_d, size);

  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

  const unsigned int numThreads = 256;
  unsigned int numBlocks = cdiv(n, numThreads);

  vecSubKernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d, n);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(C, C_d, size, cudaMemcpyHostToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}


int main() {
  const int n = 1000;
  float A[n];
  float B[n];
  float C[n];

  for (int i = 0; i < n; i += 1) {
    A[i] = float(i);
    B[i] = A[i] / 1000.0f;
  }

  // do the subtraction
  vecSub(A, B, C, n);

  // print the result
  for (int i = 0; i < n; i += 1) {
    if (i > 0) {
      printf(", ");
      if (i % 10 == 0) {
        printf("\n");
      }
    }
    printf("%8.3f", C[i]);
  }
  printf("\n");
  
  return 0;
}