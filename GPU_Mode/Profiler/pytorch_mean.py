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


"""
torch.mean() time: 1.4960806369781494
tensor.mean() time: 1.4798515105247498
manual mean time: 1.4730611181259154
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                             aten::mean        62.73%       2.403ms        63.75%       2.442ms       2.442ms       1.449ms       100.00%       1.449ms       1.449ms             1  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       1.448ms        99.97%       1.448ms       1.448ms             1  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us       0.480us         0.03%       0.480us       0.480us             1  
                                       aten::as_strided         0.09%       3.518us         0.09%       3.518us       3.518us       0.000us         0.00%       0.000us       0.000us             1  
                                        cudaMemsetAsync         0.33%      12.483us         0.33%      12.483us      12.483us       0.000us         0.00%       0.000us       0.000us             1  
                                       cudaLaunchKernel         0.60%      23.158us         0.60%      23.158us      23.158us       0.000us         0.00%       0.000us       0.000us             1  
                                  cudaDeviceSynchronize        36.25%       1.389ms        36.25%       1.389ms       1.389ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.831ms
Self CUDA time total: 1.449ms

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                             aten::mean        62.29%       2.364ms        63.04%       2.393ms       2.393ms       1.448ms       100.00%       1.448ms       1.448ms             1  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       1.448ms        99.97%       1.448ms       1.448ms             1  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us       0.416us         0.03%       0.416us       0.416us             1  
                                       aten::as_strided         0.07%       2.702us         0.07%       2.702us       2.702us       0.000us         0.00%       0.000us       0.000us             1  
                                        cudaMemsetAsync         0.28%      10.693us         0.28%      10.693us      10.693us       0.000us         0.00%       0.000us       0.000us             1  
                                       cudaLaunchKernel         0.39%      14.899us         0.39%      14.899us      14.899us       0.000us         0.00%       0.000us       0.000us             1  
                                  cudaDeviceSynchronize        36.96%       1.403ms        36.96%       1.403ms       1.403ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.795ms
Self CUDA time total: 1.448ms

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                              aten::sum        69.08%       3.150ms        69.71%       3.179ms       3.179ms       1.446ms        99.87%       1.446ms       1.446ms             1  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       1.446ms        99.85%       1.446ms       1.446ms             1  
                                              aten::div         0.37%      17.070us         0.57%      25.992us      25.992us       1.856us         0.13%       1.856us       1.856us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.856us         0.13%       1.856us       1.856us             1  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us       0.384us         0.03%       0.384us       0.384us             1  
                                       aten::as_strided         0.06%       2.637us         0.06%       2.637us       2.637us       0.000us         0.00%       0.000us       0.000us             1  
                                        cudaMemsetAsync         0.23%      10.671us         0.23%      10.671us      10.671us       0.000us         0.00%       0.000us       0.000us             1  
                                       cudaLaunchKernel         0.53%      24.287us         0.53%      24.287us      12.143us       0.000us         0.00%       0.000us       0.000us             2  
                                  cudaDeviceSynchronize        29.72%       1.355ms        29.72%       1.355ms       1.355ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.560ms
Self CUDA time total: 1.448ms
"""