# Notes on GPU-Mode lecture 1

# Profiler

Computer performance is about time and memory trade-offs. Since calculating devices are way more expensive, most of the time, time is the priority to care about.

Why use a profiler?

1. CUDA is async so can't use the Python time module
2. Profilers are way more powerful

## Tools

There are three profilers:

- autograd profiler: numerical
- Pytorch profiler: visual
- NVIDIA Nsight Compute

Autograd profiler utilizes `torch.cuda.Event()` to measure performance.

PyTorch profiler utilizes the method `profile()` from the Profiler context manager `torch.profiler` to analyze performance.
You can export the result as a `.json` file and upload it to [chrome://tracing/](chrome://tracing/) to visualize it.

## Demo

The course provides a simple program to show how to use autograd profiler to analyze the performance of three ways to do square operations:

- by `torch.square()`
- by ` **` operator
- by `*` operator

```python
def time_pytorch_function(func, input):
    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

time_pytorch_function(torch.square, b)
time_pytorch_function(square_2, b)
time_pytorch_function(square_3, b)
```

The result below is done on the NVIDIA T4 GPU.

```
Profiling torch.square:
Self CPU time total: 10.577ms
Self CUDA time total: 3.266ms

Profiling a * a:
Self CPU time total: 5.417ms
Self CUDA time total: 3.276ms

Profiling a ** 2:
Self CPU time total: 6.183ms
Self CUDA time total: 3.274ms
```

It turns out:

- CUDA operation is faster than CPU.
- The `*` operator is doing an `aten::multiply` operation rather than an `aten::pow`, and the former is faster. It is probably because that multiply is used more than pow and many developers spend time on optimizing it.
- The performance difference on CUDA is minimal. `torch.square` is the slowest operation considering the CPU time
- `aten::square` is a call to `aten::pow`
- All three methods launched a cuda kernel called `native::vectorized_elementwise_kernel<4, at...`

---

# Integrating CUDA kernels in PyTorch

There are a couple of ways to do that:

- use `load_inline` from `torch.utils.cpp_extendsion`
- use Numba which is a compiler that compiles a decorated Python function into the machine code that runs on both CPU and GPU
- use Triton

We can use `load_inline` from `torch.utils.cpp_extendsion` to load the CUDA kernel as a PyTorch extension by `load_inline(name, cpp_sources, cuda_sources, functions, with_cuda, build_directory)`.

```python
from torch.utils.cpp_extension import load_inline

square_matrix_extension = load_inline(
    name='square_matrix_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='./load_inline_cuda',
    # extra_cuda_cflags=['--expt-relaxed-constexpr']
)

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
print(square_matrix_extension.square_matrix(a))
```

---

# Hands-on

## Use autograd profiler on `mean` operation

When using autograd profiler, remember:

1. Warmup the GPU before recording so that the GPU enters a steady state
2. Average multiple runs for more reliable results

```python
import torch

# Method 1: use `torch.mean()`
def mean_all_by_torch(input_tensor):
    return torch.mean(input_tensor)

# Method 2: use `mean()` of the tensor
def mean_all_by_tensor(input_tensor):
    return input_tensor.mean()

# Method 3: use `torch.sum()` and `tensor.numel()`
def mean_all_by_combination(input_tensor):
    return torch.sum(input_tensor) / input_tensor.numel()

def time_pytorch_function(func, input_tensor, warmup=5, runs=100):
    # Warmup
    for _ in range(warmup):
      func(input_tensor)

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(runs):
        start.record()
        func(input_tensor)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)

input_tensor = torch.randn(10000, 10000).cuda()

print("torch.mean() time:", time_pytorch_function(mean_all_by_torch, input_tensor))
print("tensor.mean() time:", time_pytorch_function(mean_all_by_tensor, input_tensor))
print("manual mean time:", time_pytorch_function(mean_all_by_combination, input_tensor))


with torch.profiler.profile() as prof:
    mean_all_by_torch(input_tensor)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with torch.profiler.profile() as prof:
    mean_all_by_tensor(input_tensor)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with torch.profiler.profile() as prof:
    mean_all_by_combination(input_tensor)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Use Pytorch profiler on `mean` operation

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        mean_tensor = torch.mean(torch.randn(10000, 10000).cuda())

prof.export_chrome_trace("mean_trace.json")
```

## Implementing triton code for `torch.mean()`

```python
import triton
import triton.language as tl
import torch

@triton.jit
def mean_kernel(
    x_ptr,          # pointer to input tensor
    output_ptr,     # pointer to output tensor
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,  # number of elements per block
):

    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)

    acc = 0.0

    for idx in range(block_start, block_end):
        x = tl.load(x_ptr + idx)
        acc += x


    block_mean = acc / n_elements

    # Store result
    tl.store(output_ptr + pid, block_mean)

# Wrapper function
def triton_mean(x: torch.Tensor) -> torch.Tensor:

    x = x.contiguous().view(-1)
    n_elements = x.numel()


    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)


    output = torch.empty(grid[0], device=x.device, dtype=x.dtype)


    mean_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.sum()

# Example usage:
if __name__ == "__main__":
    # Create test tensor
    x = torch.randn(1000000, device='cuda')

    # Compare results
    torch_mean = torch.mean(x)
    triton_mean_result = triton_mean(x)

    print(f"PyTorch mean: {torch_mean}")
    print(f"Triton mean: {triton_mean_result}")
    print(f"Difference: {abs(torch_mean - triton_mean_result)}")
```

---

# Reference

- [gpu-mode lectures - Github](https://github.com/gpu-mode/lectures)
- [Event - Pytorch](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [torch.utils.cpp_extension.load_inline](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline)
- [Triton](https://triton-lang.org/main/index.html)
