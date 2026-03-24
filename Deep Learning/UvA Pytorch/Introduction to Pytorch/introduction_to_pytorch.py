import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to PyTorch
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [**PyTorch**](https://pytorch.org) is an **open-source** machine learning framework that allows developers and researchers to write neural networks and optimize them efficiently.

    PyTorch has a huge developer community (originally developed by Facebook). Many current papers publish their code in PyTorch.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Installation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first thing is to **install PyTorch and import it**.

    You may refer to the [locally section from PyTorch official documentation](https://pytorch.org/get-started/locally/) to install the **latest version**.

    If you intend to install a **previous version**(e.g., your project relies on legacy libraries), check [Installing previous versions of PyTorch](https://pytorch.org/get-started/previous-versions/).
    """)
    return


@app.cell
def _():
    # Standard libraries
    import os
    import math
    import numpy as np
    import time
    import torch
    import marimo as mo
    import ipywidgets as widgets
    from IPython.display import display

    # Imports for plotting
    import matplotlib.pyplot as plt
    # `%matplotlib inline` marimo auto-displays plots. See: [Coming from Jupyter](https://docs.marimo.io/guides/coming_from/jupyter/#working-with-expensive-notebooks)
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    # For export
    set_matplotlib_formats('svg', 'pdf')
    from matplotlib.colors import to_rgba

    import seaborn as sns
    sns.set()

    # Progress bar
    from tqdm.notebook import tqdm

    return mo, np, time, torch, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The PyTorch package is called `torch`.

    Get the version by `torch.__version__`
    """)
    return


@app.cell
def _(torch):
    print(f"Using torch version:", torch.__version__)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set the seed
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PyTorch provides functions that are stochastic, like generating random numbers.

    A **good practice** is to make the code **reproducible** by setting the **exact same random numbers** via [`torch.manual_seed()`](https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html).

    Note: You may find that many code use the number `42` for the seed. The answer is in _The Hitchhiker’s Guide to the Galaxy_.
    """)
    return


@app.cell
def _(torch):
    torch.manual_seed(42)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tensors
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) are similar to [**numpy arrays**](https://numpy.org/doc/stable/reference/generated/numpy.array.html), and many operations from numpy can be used on tensors.

    [Tensor is a **generalization** of concepts from **matrices and vectors**.](https://en.wikipedia.org/wiki/Tensor)
      * A vector is a 1D tensor
      * A matrix is a 2D tensor
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Initialization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are [many methods to create tensors](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor):
    * **`torch.tensor()`**: create a tensor with **pre-exisiting data**.
    * **`torch.zeros()`**: create a tensor by the given shape filled with **`0`**.
    * **`torch.ones()`**: create a tensor by the given shape filled with **`1`**.
    * **`torch.rand()`**: create a tensor by the given shape filled with **random values uniformly sampled between `0` and `1`**.
    * **`torch.randn()`**: create a tensor by the given shape filled with **random values sampled from a normal distribution with mean `0` and variance `1`**.
    * **`torch.arange(start, end)`**: create a tensor containing the **values from `start` to `end`(not included)**.
    * **`torch.*_like()`**: create a tensor with **the same size (and similar types)** as another tensor.
    * **`torch.new_()`**: create a tensor with **a similar type but different size** from another tensor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `torch.Tensor()` **allocates memory** for the tensor and **reuses** any values that have already been in the memory.

    > Note: [There is a legacy constructor torch.Tensor whose use is discouraged. Use `torch.tensor()` instead.](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
    """)
    return


@app.cell
def _(torch):
    tensor_with_given_values = torch.tensor([[1, 2], [3, 4]])
    tensor_with_zeros = torch.zeros((1, 2, 3))
    tensor_with_ones = torch.ones(1, 2, 3)
    tensor_with_random = torch.rand(1, 2, 3)
    tensor_with_random_normal = torch.randn(1, 2, 3)
    tensor_with_arange = torch.arange(start=1, end=10)


    print(tensor_with_given_values)
    print("-------------------------")
    print(tensor_with_zeros)
    print("-------------------------")
    print(tensor_with_ones)
    print("-------------------------")
    print(tensor_with_random)
    print("-------------------------")
    print(tensor_with_random_normal)
    print("-------------------------")
    print(tensor_with_arange)
    print("-------------------------")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Shape
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Obtain the **shape** of a tensor by the [`.shape`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.shape.html#torch.Tensor.shape) attribute or the [`.size()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.size.html#torch.Tensor.size) method.
    """)
    return


@app.cell
def _(torch):
    x_tensor = torch.empty(1, 2, 3)
    print("The shape of tensor x_tensor: ", x_tensor.shape)
    print("The shape of tensor x_tensor: ", x_tensor.size())

    dim1, dim2, dim3 = x_tensor.shape
    print(f"dimension1: {dim1}")
    print(f"dimension1: {dim2}")
    print(f"dimension1: {dim3}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tensor to numpy and vice visa
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tensor to numpy: [`.numpy()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html#torch.Tensor.numpy).
    * In order to convert a tensor to a numpy array, the tensor needs to be on the **CPU**. So `np_arr = tensor.cpu().numpy()`.

    Numpy to tensor: [`torch.from_numpy()`](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html#torch.from_numpy.)
    """)
    return


@app.cell
def _(np, torch):
    # Tensor to numpy
    tensor = torch.Tensor(1, 2, 3)
    print(f"type: {type(tensor)}")
    tensor_to_numpy = tensor.numpy()
    print(f"type: {type(tensor_to_numpy)}")

    # numpy to tensor
    np_array = np.array([1, 2, 3])
    print(f"type: {type(np_array)}")
    numpy_to_torch = torch.from_numpy(np_array)
    print(f"type: {type(numpy_to_torch)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Operations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Add and reshape
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * **Add** two tensors
        * creating a **new** tensor: **`+`**.
        * **In-place add**: by [**`Tensor.add_(other)`**](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.add_.html#torch.Tensor.add_).
    * **Reshape** a tensor with the same values: **`.view()`**.

    **In-place ops** are usually marked with **underscore postfix**, eg, **`add_`**.
    """)
    return


@app.cell
def _(torch):
    # Add by creating a new tensor
    x1 = torch.randn(1, 2)
    x2 = torch.randn(1, 2)
    y = x1 + x2
    print("x1:", x1)
    print("x2:", x2)
    print("y:", y)
    print(x1+x2 == y)

    # In-place add
    print(f"x2: {x2}")
    print("x1 + x2", x2.add_(x1))

    # Re-shape
    x_original = torch.arange(1, 10)
    x_reshaped = x_original.view(3, 3)
    print(f"x_original: {x_original}")
    print(f"x_reshaped: {x_reshaped}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Matrix multiplications

    * [**`torch.matmul()`**](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html#torch-matmul):
      * same to **`a @ b`**
      * support **broadcasting**
    * [**`torch.mm()`**](https://docs.pytorch.org/docs/stable/generated/torch.mm.html#torch.mm)
      * doesn't support broadcasting
    * [**`torch.bmm()`**](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html#torch-bmm)
      * perform **batch matrix multiplication**
      * $T(\text{shape: } b \times n \times m) \times R(\text{shape: } b \times m \times p) = O(\text{shape: } b \times n \times p)$
      * doesn't support broadcasting
    * [**`torch.einsum()`**](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html#torch-einsum)
      * support matrix multiplication and more (i.e., sums of products) using the Einstein summation convention.

    Usually, we use `torch.matmul()` or `torch.bmm()`.
    """)
    return


@app.cell
def _(torch):
    x = torch.arange(6)
    x = x.view(2, 3)
    x_3d = torch.arange(8).view(2, 2, 2)
    print("X:", x)
    print("X_3d:", x_3d)

    W = torch.arange(9).view(3, 3)
    W_3d = torch.arange(8).view(2, 2, 2)
    print("W:", W)
    print("W_3d:", W_3d)

    output_by_matmul = torch.matmul(x, W)
    output_by_mm = torch.mm(x, W)
    output_by_einsum = torch.einsum('ij,jk->ik', x, W)
    output_by_bmm = torch.bmm(x_3d, W_3d)
    print("output_by_matmul", output_by_matmul)
    print("output_by_mm", output_by_mm)
    print("output_by_einsum", output_by_einsum)
    print(output_by_matmul == output_by_mm)
    print(output_by_mm == output_by_einsum)
    print("output_by_bmm", output_by_bmm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Indexing & Slicing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Selecting a part of a tensor by `:`.
    """)
    return


@app.cell
def _(torch):
    x_indexing = torch.arange(12).view(3, 4)
    print(f"x_indexing: {x_indexing}")
    print("First row", x_indexing[0,:])
    print("First column", x_indexing[:,0])
    print("last row", x_indexing[-1,:])
    print("last column", x_indexing[:,-1])
    print("Element at first row and second column:", x_indexing[0,1])
    print("Element at second row and first column:", x_indexing[1,0])
    print("First two rows and second column:", x_indexing[:2,1:2])
    print("First row and the last column:", x_indexing[0,-1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dynamic computation graph and backpropagation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The only thing developers have to do is to **compute the output**, and PyTorch can **automatically get the gradients**.
    * PyTorch can **automatically** get **gradients/derivatives of functions**.
    * Weight matrices that could learn are called the **parameters** or the **weights**.
    * If the neural network outputs **a single scalar value**, it's the **derivative**, but quite often we will have **multiple output variables**; in that case it's the **gradients**.
    * As we manipulate the input, we are automatically creating a **computational graph**, which shows **how to get the output from the input**.
    * PyTorch is a **define-by-run** framework, which means that developers can just do manipulations, and PyTorch will keep track of that graph.

    Want to learn more?

    Read:
    * [Automatic Differentiation with `torch.autograd`](**https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html**)
    * [A Gentle Introduction to `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
    * [The Fundamentals of Autograd](https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html)
    * [Computational Graphs, and Backpropagation from Columbia University](https://www.cs.columbia.edu/~mcollins/ff2.pdf)
    * [Overview of PyTorch Autograd Engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/)
    * [How Computational Graphs are Constructed in PyTorch](https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/)
    * [How Computational Graphs are Executed in PyTorch](https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first thing is **specify which tensor requires gradients**.

    By default, the tensor created does **not** require gradients.
    """)
    return


@app.cell
def _(torch):
    tensor_without_gradients = torch.ones((3,))
    print(tensor_without_gradients)
    print(tensor_without_gradients.requires_grad) # False
    print(tensor_without_gradients.grad) # None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using [torch.Tensor.requires_grad()](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html) to specify that the tensor requires gradients.

    Only tensor with **float** type value can have gradients.
    """)
    return


@app.cell
def _(torch):
    tensor_with_gradients = torch.ones((3,)).requires_grad_(True)
    print(tensor_with_gradients)
    print(tensor_with_gradients.requires_grad) # True
    print(tensor_with_gradients.grad) # None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
      <mi>y</mi>
      <mo>=</mo>
      <mfrac>
        <mn>1</mn>
        <mrow>
          <mi>&#x2113;</mi>
          <mo stretchy="false">(</mo>
          <mi>x</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </mfrac>
      <munder>
        <mo data-mjx-texclass="OP">&#x2211;</mo>
        <mi>i</mi>
      </munder>
      <mrow data-mjx-texclass="INNER">
        <mo data-mjx-texclass="OPEN">[</mo>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>x</mi>
          <mi>i</mi>
        </msub>
        <mo>+</mo>
        <mn>2</mn>
        <msup>
          <mo stretchy="false">)</mo>
          <mn>2</mn>
        </msup>
        <mo>+</mo>
        <mn>3</mn>
        <mo data-mjx-texclass="CLOSE">]</mo>
      </mrow>
      <mo>,</mo>
    </math>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The computation graph:
    """)
    return


@app.cell
def _(torch):
    x_input = torch.arange(3, dtype=torch.float32, requires_grad=True)
    print(x_input)
    a = x_input + 2
    b = a ** 2
    c = b + 3
    output = c.mean()
    print(output)
    return output, x_input


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * `grad_fn()`: Each node in the computation graph automatically defines a function for calculating the gradients with respect to its inputs, `grad_fn`.
    * `backward()`: Calling the function [`backward()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html) on the last output to perform backpropagation to get the gradients of each tensor that has `requires_grad = True`
    """)
    return


@app.cell
def _(output):
    # perform backpropagation
    output.backward()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <mi>&#x2202;</mi>
      <mi>y</mi>
      <mrow data-mjx-texclass="ORD">
        <mo>/</mo>
      </mrow>
      <mi>&#x2202;</mi>
      <mrow data-mjx-texclass="ORD">
        <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">x</mi>
      </mrow>
    </math>
    """)
    return


@app.cell
def _(x_input):
    print(f"The gradient of x_input: {x_input.grad}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GPU support
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For macOS, one should use [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html).
    """)
    return


@app.cell
def _(torch):
    gpu_avail_mac = torch.backends.mps.is_available()
    print(f"Is the GPU available on Mac? {gpu_avail_mac}")
    return


@app.cell
def _(torch):
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"Device: {device}")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By **default**, tensor created are on the **CPU**.

    Move tensors to GPU by `.to()` or `.cuda()`.

    A **good practice** is to use a variable(e.g., `device`) to point to the device to use so that we can run the same code on both CPU and GPU.
    """)
    return


@app.cell
def _(device, torch):
    x_on_gpu = torch.zeros(2, 3)
    x_on_gpu = x_on_gpu.to(device)

    print("X on GPU:", x_on_gpu)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can see the `device='mps:0'` from the above output, which means it's the `0-th` GPU on the device.

    PyTorch supports multi-GPU system.
    """)
    return


@app.cell
def _(device, time, torch):
    # performance benchmark
    x_cpu = torch.randn(5000, 5000)

    # CPU version
    print("\n--- CPU Benchmark ---")
    start_time_cpu = time.time()
    _cpu = torch.matmul(x_cpu, x_cpu)
    end_time_cpu = time.time()
    print(f"CPU time: {(end_time_cpu - start_time_cpu):6.5f}s")

    # GPU version
    print("\n--- GPU Benchmark ---")
    x_gpu = x_cpu.to(device)
    # warm up
    warm_up = torch.matmul(x_gpu, x_gpu)
    torch.mps.synchronize() # waits for MPS operations to finish

    start_time_gpu = time.time()
    _gpu = torch.matmul(x_gpu, x_gpu)
    torch.mps.synchronize()
    end_time_gpu = time.time()
    print(f"GPU time: {(end_time_gpu - start_time_gpu):6.5f}s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When generating random numbers, the **seed** between CPU and GPU is **not synchronized**.

    To ensure **reproducibility**, set the seed on the GPU separately.

    Note: Due to **different** GPU architectures, the random numbers generated by the same code may **not be the same**.

    Some operations on a GPU are implemented **stochastically** for effiency. To ensure reproducibility, make these operations **deterministic** by [`torch.use_deterministic_algorithms(True)`](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
    """)
    return


@app.cell
def _(torch):
    # Set the seed for CPU
    torch.manual_seed(0)

    # For MPS
    if torch.backends.mps.is_available():
        print("MPS device found")
        torch.mps.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        print("MPS manual seed set. Deterministic algorithms requested")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # An example: Continuos XOR
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A simple neuron(i.e., ) **cannot** learn the XOR opeartion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `torch.nn` package defines the **building blocks** of neural networks.

    `torch.nn.functional` package defines the **functionalities** used in the neural networks.
    """)
    return


@app.cell
def _():
    import torch.nn as nn
    import torch.nn.functional as F

    return (nn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A neural network is built up out of **modules(`nn.Module`)** and implemented as a `class` object.

    * In `__init__` method: defines the layers using `nn.Parameter`
    * In `forward()` method: defines the forward pass(computation)
    * The backward pass is done **automatically**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The model class
    """)
    return


@app.cell
def _(nn):
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            pass

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A simple classifer.
    """)
    return


@app.cell
def _(nn):
    class SimpleClassifier(nn.Module):
        def __init__(self, num_inputs, num_hidden, num_outputs):
            super().__init__()
            self.linear1 = nn.Linear(num_inputs, num_hidden)
            self.act_fn = nn.Tanh()
            self.linear2 = nn.Linear(num_hidden, num_outputs)

        def forward(self, x):
            x = self.linear1(x)
            x = self.act_fn(x)
            x = self.linear2(x)
            return x

    return (SimpleClassifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Initialize the model.
    """)
    return


@app.cell
def _(SimpleClassifier):
    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    print(model)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Print the model and list all the submodules.

    * The **parameters** of a module could be **obtained** by `parameters()` or `named_parameters()` to get a name to each parameter object.
    * The activation function **doesn't** contains parameters.
    * The parameters are only registered for `nn.Module` objects that are direct object attributes. Define **a list of modules**, consider using `nn.ModuleList`, `nn.ModuleDict` and `nn.Sequential`
    """)
    return


@app.cell
def _(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, shape: {param.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PyTorch uses `torch.utils.data` to provide functionalities to load training and test data.

    The data package provides two classes:
    * `data.Dataset`: an interface to **access the training and test data**
    * `data.DataLoader`: load and stack the data points from dataset into batches.
    """)
    return


@app.cell
def _():
    import torch.utils.data as data

    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The dataset class
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To define a dataset in PyTorch, simply specify two functions: `__getitem__` and `__len__`
    * `__getitem__` returns the `i-th` data point in the dataset
    * `__len__` returns the **size** of the dataset
    """)
    return


@app.cell
def _(data, torch):
    class XORDataset(data.Dataset):
        def __init__(self, size, std=0.1):
            super().__init__()
            self.size = size
            self.std = std
            self.generate_continuous_xor()

        def generate_continuous_xor(self):
            data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
            label = (data.sum(dim=1) == 1).to(torch.long)
            data += self.std * torch.randn(data.shape)

            self.data = data
            self.label = label

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            data_point = self.data[idx]
            data_label = self.label[idx]
            return data_point, data_label

    return (XORDataset,)


@app.cell
def _(XORDataset):
    dataset = XORDataset(size=200)
    print("Size of dataset:", len(dataset))
    print("Data point at index 0:", dataset[0])
    return (dataset,)


@app.cell
def _():
    import plotly.express as px

    return (px,)


@app.cell
def _(px, torch):
    def visualize_samples(data, label):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        class_labels = ["Class 0" if l == 0 else "Class 1" for l in label]

        fig = px.scatter(
            x=data[:,0],
            y=data[:,1],
            color=class_labels,
            title="Dataset samples",
            labels={
                "x": "$x_1$",
                "y": "$x_2$",
                "color": "Classes"
            },
            width=500,
            height=500
        )

        fig.update_traces(marker=dict(line=dict(width=1, color="#333")))
        fig.show()

    return (visualize_samples,)


@app.cell
def _(dataset, visualize_samples):
    visualize_samples(dataset.data, dataset.label)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The dataloader class
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The class `torch.utils.data.DataLoader` is a Python iterable over a dataset with support for automatic batching, multi-process data loading, and other features.
    * It communicate with dataset via `__getitem__` and stacks the outputs as tensors over the first dimension to form a batch.
    * Some configuration parameters:
        * `batch_size`: Number of samples to stack per batch.
        * `shuffle`: If `True`, the data is returned in a **random order**. It's important for stochasticity in training.
        * `num_workers`: Number of subprocesses to use for data loading.
          * The default is `0`, which means that the data will be loaded in the main process, which can slow down the training since loading data also requires time.
          * More workers are recommended for larger datasets. For tiny datasets, `0` worker would be sufficient.
        * `pin_memory`: If `True`, the data loader will copy tensors into CUDA pinned memory before returning.
          * It can save time for large data points on GPU.
          * It's a good practice for training data but not necessary for validation and testing data.
        * `drop_last`: If `True`, the last batch will be dropped if it is not complete.
          * It's useful when the dataset size is **not divisable** by the batch size.
    """)
    return


@app.cell
def _(data, dataset):
    data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)
    return (data_loader,)


@app.cell
def _(data_loader):
    data_inputs, data_labels = next(iter(data_loader))
    print("Data inputs", data_inputs.shape, "\n", data_inputs)
    print("Data labels", data_labels.shape, "\n", data_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The optimization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    During training, we usually perform the following steps:
    * Get a batch from the data loader
    * Obtain the predictions from the model for the batch
    * Calculate the loss based on the difference between predictions and labels
    * Backpropagation: calculate the gradients for every parameter with respect to the loss
    * Update the parameters of the model in the direction of the gradients
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The loss function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For binary classification, we use **Binary Cross Entropy(BCE)** to calculate the loss.

    <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
      <msub>
        <mrow data-mjx-texclass="ORD">
          <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">L</mi>
        </mrow>
        <mrow data-mjx-texclass="ORD">
          <mi>B</mi>
          <mi>C</mi>
          <mi>E</mi>
        </mrow>
      </msub>
      <mo>=</mo>
      <mo>&#x2212;</mo>
      <munder>
        <mo data-mjx-texclass="OP">&#x2211;</mo>
        <mi>i</mi>
      </munder>
      <mrow data-mjx-texclass="INNER">
        <mo data-mjx-texclass="OPEN">[</mo>
        <msub>
          <mi>y</mi>
          <mi>i</mi>
        </msub>
        <mi>log</mi>
        <mo data-mjx-texclass="NONE">&#x2061;</mo>
        <msub>
          <mi>x</mi>
          <mi>i</mi>
        </msub>
        <mo>+</mo>
        <mo stretchy="false">(</mo>
        <mn>1</mn>
        <mo>&#x2212;</mo>
        <msub>
          <mi>y</mi>
          <mi>i</mi>
        </msub>
        <mo stretchy="false">)</mo>
        <mi>log</mi>
        <mo data-mjx-texclass="NONE">&#x2061;</mo>
        <mo stretchy="false">(</mo>
        <mn>1</mn>
        <mo>&#x2212;</mo>
        <msub>
          <mi>x</mi>
          <mi>i</mi>
        </msub>
        <mo stretchy="false">)</mo>
        <mo data-mjx-texclass="CLOSE">]</mo>
      </mrow>
    </math>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PyTorch provides a list of loss functions. For BCE, there are [`nn.BCELoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) and [`nn.BCEWithLogitsLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
    * The `nn.BCELoss()` requires the inputs to be in the range `[0, 1]`, i.e., the output of a sigmoid function.
    * The `nn.BCEWithLogitsLoss()` combines a sigmoid layer and the BCE loss in a single class. Thus, it is **numerically more stable**.
    * Hence, it is advised to **use loss functions applied on "logits" where possible**.
    """)
    return


@app.cell
def _(nn):
    loss_module = nn.BCEWithLogitsLoss()
    return (loss_module,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stochastic Gradient Descent(SGD)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PyTorch provides the package [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html) that implements various optimization algorithms.

    SGD updates parameters by multiplying the gradient by a learning rate, and subtracting those from the parameters(hence minimizing the loss).
    """)
    return


@app.cell
def _(model, torch):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return (optimizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `optimizer` provides:
    * `optimizer.step()`: updates the parameters based on the gradients.
    * `optimizer.zero_grad()`: sets the gradients of all parameters to zero, which is a crucial pre-step before performing backpropagation.

    Remember to call `optimizer.zero_grad()` before calculating the gradients of a batch. Otherwise, if we call the `backward` function without zeroing the gradients, it will be added to the previous ones instead of overwriting them. This is because a parameter occurs multiple times in a computation graph, and we need to sum the gradients in this case instead of replacing them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training
    """)
    return


@app.cell
def _(XORDataset, data):
    train_dataset = XORDataset(size=2500)
    train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    return (train_data_loader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this (tiny) task, pushing the model and data into the GPU device actually takes more time than training on the CPU. But for a larger network, the speedup can be significant.
    """)
    return


@app.cell
def _(device, model):
    model.to(device=device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make sure to set the model to **training mode** by calling **`model.train()`**.

    For evaluation, set the model to **evaluation mode** by calling **`model.eval()`**.
    """)
    return


@app.cell
def _(device, tqdm):
    def train_model(model, optimizer, data_loader, loss_module, num_epoches=100):
        # Set the model to train mode
        model.train()

        # Training loop
        for epoch in tqdm(range(num_epoches)):
            for data_inputs, data_labels in data_loader:
                # Step 1: Move input data to device
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)

                # Step 2: Run the model on the input data
                preds = model(data_inputs)
                # Output has shape `[batch_size, 1]`, but we want `[batch_size]`
                preds = preds.squeeze(dim=1)

                # Step 3: Calculate the loss
                loss = loss_module(preds, data_labels.float())

                # Step 4: Perform backpropagation
                # Before calculating the gradients, we need to ensure that they are all zero.
                # The gradients would not be overwritten, but actually added to the existing ones.
                optimizer.zero_grad()
                # Perform backpropagation
                loss.backward()


                optimizer.step()

    return (train_model,)


@app.cell
def _(loss_module, model, optimizer, train_data_loader, train_model):
    train_model(model, optimizer, train_data_loader, loss_module)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving a model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can save the trained model to disk to load the weights later by using extracting the **`state_dict()`** of the model and using **`torch.save()`** to save it.
    """)
    return


@app.cell
def _(model):
    state_dict = model.state_dict()
    print(state_dict)
    return (state_dict,)


@app.cell
def _(state_dict, torch):
    torch.save(state_dict, "model.tar")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to load a saved model, we need to create a model instance first and use **`load_state_dict(torch.load())`** to load the weights.
    """)
    return


@app.cell
def _(SimpleClassifier, model, torch):
    new_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    new_model.load_state_dict(torch.load('model.tar'))

    print("Original model\n", model.state_dict())
    print("\nLoaded model\n", new_model.state_dict())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As metric, we will use accuracy.
    <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
      <mi>a</mi>
      <mi>c</mi>
      <mi>c</mi>
      <mo>=</mo>
      <mfrac>
        <mrow>
          <mi mathvariant="normal">#</mi>
          <mtext>correct predictions</mtext>
        </mrow>
        <mrow>
          <mi mathvariant="normal">#</mi>
          <mtext>all predictions</mtext>
        </mrow>
      </mfrac>
      <mo>=</mo>
      <mfrac>
        <mrow>
          <mi>T</mi>
          <mi>P</mi>
          <mo>+</mo>
          <mi>T</mi>
          <mi>N</mi>
        </mrow>
        <mrow>
          <mi>T</mi>
          <mi>P</mi>
          <mo>+</mo>
          <mi>T</mi>
          <mi>N</mi>
          <mo>+</mo>
          <mi>F</mi>
          <mi>P</mi>
          <mo>+</mo>
          <mi>F</mi>
          <mi>N</mi>
        </mrow>
      </mfrac>
    </math>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Don't forget to set the model to eval mode by **`model.eval()`**.

    For **testing**, there is **no need to calculate gradients** and keep track of the computation graph, so using **`with torch.no_grad():`** to deactivate the gradient calculation.
    """)
    return


@app.cell
def _(XORDataset, data):
    test_dataset = XORDataset(size=500)
    test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    return (test_data_loader,)


@app.cell
def _(device, torch):
    def eval_model(model, data_loader):
        # set model to eval mode
        model.eval()
        true_preds, num_preds = 0., 0

        with torch.no_grad():
            for data_inputs, data_labels in data_loader:
                data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
                preds = model(data_inputs)
                preds = preds.squeeze(dim=1)
                preds = torch.sigmoid(preds)
                pred_labels = (preds >= 0.5).long()

                true_preds += (pred_labels == data_labels).sum()
                num_preds += data_labels.shape[0]

            acc = true_preds / num_preds
            print(f"Accuracy of the model: {100.0*acc:4.2f}%")

    return (eval_model,)


@app.cell
def _(eval_model, model, test_data_loader):
    eval_model(model, test_data_loader)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualizing classification boundaries
    """)
    return


@app.cell
def _(dataset, device, model, torch):
    import plotly.graph_objects as go

    @torch.no_grad()
    def visualize_classification(model, data, label):
        # Convert data to numpy
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        data_0 = data[label == 0]
        data_1 = data[label == 1]

        # --- 1. Get Model Predictions for Background ---
        model.to(device)

        # Create coordinate grid
        x_range = torch.arange(-0.5, 1.5, step=0.01, device=device)
        y_range = torch.arange(-0.5, 1.5, step=0.01, device=device)

        xx1, xx2 = torch.meshgrid(x_range, y_range, indexing="ij")
        model_inputs = torch.stack([xx1, xx2], dim=-1)

        preds = model(model_inputs)
        preds = torch.sigmoid(preds).squeeze()

        # Convert to numpy. 
        # NOTE: Because we used indexing="ij", the shape is (x, y). 
        # Plotly expects (y, x) for heatmaps, so we Transpose (.T) it!
        preds_np = preds.cpu().numpy().T 

        # --- 2. Build the Plotly Figure ---
        fig = go.Figure()

        # Add background decision boundary
        fig.add_trace(go.Heatmap(
            x=x_range.cpu().numpy(),
            y=y_range.cpu().numpy(),
            z=preds_np, 
        
            colorscale=[[0, '#1f77b4'], [1, '#ff7f0e']], 
            zmin=0, zmax=1,
            showscale=False,    # Hide the colorbar
            hoverinfo="skip",   # Don't show tooltips for the background
            opacity=0.6         # Slight transparency makes points pop out more
        ))

        # Add Class 0 points
        fig.add_trace(go.Scatter(
            x=data_0[:,0], y=data_0[:,1],
            mode='markers', name='Class 0',
            marker=dict(color='#1f77b4', size=8, line=dict(color='#333', width=1))
        ))

        # Add Class 1 points
        fig.add_trace(go.Scatter(
            x=data_1[:,0], y=data_1[:,1],
            mode='markers', name='Class 1',
            marker=dict(color='#ff7f0e', size=8, line=dict(color='#333', width=1))
        ))

        # --- 3. Format Layout ---
        fig.update_layout(
            title="Dataset samples & Decision Boundary",
            xaxis_title="$x_1$",
            yaxis_title="$x_2$",
            width=500, height=500,
            # Force strict axis limits and a 1:1 aspect ratio
            xaxis=dict(range=[-0.5, 1.5], constrain="domain"),
            yaxis=dict(range=[-0.5, 1.5], scaleanchor="x", scaleratio=1),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        return fig

    # Usage:
    fig = visualize_classification(model, dataset.data, dataset.label)
    fig.show()
    return


if __name__ == "__main__":
    app.run()
