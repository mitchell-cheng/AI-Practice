import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        mean_tensor = torch.mean(torch.randn(10000, 10000).cuda())

prof.export_chrome_trace("mean_trace.json")

# Upload the "mean_trace.json" to chrome://tracing