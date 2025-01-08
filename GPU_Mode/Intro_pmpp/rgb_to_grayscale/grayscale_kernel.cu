#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__
void rgb_to_grayscale_kernel(unsigned char* output, unsigned char* input, int width, int height) {
  const int channels = 3;

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    int outputOffset = row * width + col;
    int inputOffset = (row * width + col) * channels;

    unsigned char r = input[inputOffset + 0];
    unsigned char g = input[inputOffset + 1];
    unsigned char b = input[inputOffset + 2];

    output[outputOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

torch::Tensor rgb_to_grayscale_kernel(torch::Tensor image) {
  assert(image.device().type() == torch.kCUDA);
  assert(image.dtype() == torch::kByte);

  const auto height = image.size(0);
  const auto width = image.size(1);

  auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(image.device()));

  dim3 threads_per_block(16, 16);
  dim3 number_of_blocks(cdiv(width, threads_per_block.x),
                        cdiv(height, threads_per_block.y));
  
  rgb_to_grayscale_kernel
}