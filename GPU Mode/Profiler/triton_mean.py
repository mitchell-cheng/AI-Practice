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
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute block start/end indices
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)
    
    # Initialize accumulator
    acc = 0.0
    
    # Load and sum elements in current block
    for idx in range(block_start, block_end):
        x = tl.load(x_ptr + idx)
        acc += x
    
    # Compute partial mean for this block
    block_mean = acc / n_elements
    
    # Store result
    tl.store(output_ptr + pid, block_mean)

# Wrapper function
def triton_mean(x: torch.Tensor) -> torch.Tensor:
    # Ensure input is contiguous and flattened
    x = x.contiguous().view(-1)
    n_elements = x.numel()
    
    # Define block size and grid
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Allocate output tensor
    output = torch.empty(grid[0], device=x.device, dtype=x.dtype)
    
    # Launch kernel
    mean_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Sum partial results and compute final mean
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


"""
PyTorch mean: 0.0005773120792582631
Triton mean: 0.0005773113225586712
Difference: 7.566995918750763e-10
"""