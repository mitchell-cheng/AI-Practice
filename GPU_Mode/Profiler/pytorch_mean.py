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