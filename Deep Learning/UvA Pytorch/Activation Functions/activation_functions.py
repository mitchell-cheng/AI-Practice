import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo._runtime.context.get_context().marimo_config["runtime"][
        "output_max_bytes"
    ] = 10000000000
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Activation Functions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Activation functions bring **non-linearity** to neural networks.

    Different activation functions would have different effects on models. So choose carefully.
    """)
    return


@app.cell
def _():
    # --- Standard libraries ---
    import os
    import json
    import math
    import numpy as np

    # --- Plotting ---
    from scipy.stats import gaussian_kde
    import plotly
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    # --- Progress Bar ---
    from tqdm.notebook import tqdm

    # --- PyTorch ---
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.data as data
    import torch.optim as optim

    return (
        F,
        data,
        gaussian_kde,
        go,
        json,
        make_subplots,
        math,
        nn,
        np,
        optim,
        os,
        px,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define a function to **set the seed** for all libraries used in this notebook for **reproducibility**.

    Additionally, define two **static variables** to store the **data** and the saved **model** paths.

    Its **recommended** to store **all the datasets** from PyTorch in **one joint directory** to prevent duplicate downloads.
    """)
    return


@app.cell
def _(np, torch):
    # Set the path for the dataset
    DATASET_PATH = "./data"
    # Set the path for saved model weights
    CHECKPOINT_PATH = "./saved_models"


    # Seed function for NumPy and PyTorch
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
            torch.use_deterministic_algorithms(True)


    set_seed(42)

    # Set the device
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Device", device)
    return CHECKPOINT_PATH, DATASET_PATH, device, set_seed


@app.cell
def _(CHECKPOINT_PATH, os):
    import urllib.request
    from urllib.error import HTTPError

    base_url = (
        "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial3/"
    )
    pretrained_files = [
        "FashionMNIST_elu.config",
        "FashionMNIST_elu.tar",
        "FashionMNIST_leakyrelu.config",
        "FashionMNIST_leakyrelu.tar",
        "FashionMNIST_relu.config",
        "FashionMNIST_relu.tar",
        "FashionMNIST_sigmoid.config",
        "FashionMNIST_sigmoid.tar",
        "FashionMNIST_swish.config",
        "FashionMNIST_swish.tar",
        "FashionMNIST_tanh.config",
        "FashionMNIST_tanh.tar",
    ]

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url} ...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print(f"Failed to download {file_url}. Error: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Common Activation Functions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most of the **activation functions** could be found in the [`torch.nn`](https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) package.

    Define a base class that inherits from `nn.Module` so that we can integrate them into the network. We will use the `config` directory to store adjustable parameters for some activations.
    """)
    return


@app.cell
def _(nn):
    class ActivationFunction(nn.Module):
        def __init__(self):
            super().__init__()
            self.name = self.__class__.__name__
            self.config = {"name": self.name}

    return (ActivationFunction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The oldest four are:
    * `Heaviside step`
    * `Sine`
    * `Sigmoid`
    * `Tanh`

    But they are still used in some tasks.

    The last three can be found in `torch.sin`, `torch.sigmoid`, and `torch.tanh` or as modules(`nn.Sigmoid`, `nn.Tanh`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The [Heaviside step function](https://en.wikipedia.org/wiki/Heaviside_step_function):

    $$
    f(x) = \begin{cases}
        1 & x > 0 \\\\
        0 & x \le 0
    \end{cases}
    $$

    The [Sine function](https://en.wikipedia.org/wiki/Sine_and_cosine):

    $$
    f(x) = sin(x)
    $$

    The [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function):

    $$
    f(x) = \frac{1}{1+e^{-x}}
    $$

    The [Tanh function](https://en.wikipedia.org/wiki/Hyperbolic_functions):

    $$
    f(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
    $$
    """)
    return


@app.cell
def _(ActivationFunction, math, torch):
    # Heaviside step
    class Heaviside(ActivationFunction):
        def forward(self, x):
            return torch.where(x >= 0, 1.0, 0.0)


    # Sin
    # We use Taylor Series Expansion
    class Sine(ActivationFunction):
        def forward(self, x, terms=10):
            # Initialize the sum
            series_sum = torch.zeros_like(x)

            for n in range(terms):
                numerator = ((-1) ** n) * torch.pow(x, 2 * n + 1)
                denominator = math.factorial(2 * n + 1)
                term = numerator / denominator
                series_sum += term

            return series_sum


    # Sigmoid
    class Sigmoid(ActivationFunction):
        def forward(self, x):
            return 1 / (1 + torch.exp(-x))


    # Tanh
    class Tanh(ActivationFunction):
        def forward(self, x):
            numerator = torch.exp(x) - torch.exp(-x)
            denominator = torch.exp(x) + torch.exp(-x)
            return numerator / denominator

    return Heaviside, Sigmoid, Sine, Tanh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Another popular one is **Rectified Linear Unit(ReLU)**, which allows **training deep** neural networks.

    Despite its simplicity of being a [piecewise linear function](https://en.wikipedia.org/wiki/Piecewise_linear_function), ReLU has a **strong, stable gradient for a large range of input values** compared to sigmoid and tanh. Based on this idea, a lot of **"ReLU"-like** functions were proposed, such as LeakyReLU, ELU, softplus, Swish, and GELU.
    * **LeakyReLU**: **replaces the zero** settings in the **negative** part with a smaller slope to allow gradients to flow.
    * **ELU**: replaces the negative part with **an exponential decay**.
    * **softplus**: behaves **linearly** for a large argument $x\gg1$ and **vanishes exponentially** for a negative argument. **The `softplus` does not pass through the origin.** The derivative of softplus is sigmoid.
    * **Swish**: is both **smooth** and **non-monotonic**, which **prevents dead neurons** as in standard ReLU. See [Searching for Activation Functions](https://arxiv.org/abs/1710.05941).
    * **GELU(The Gaussian Error Linear Unit)**: is a lot like the Swish.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The [ReLU function](https://en.wikipedia.org/wiki/Rectified_linear_unit):

    $$
    f(x) = \begin{cases}
        x & x > 0 \\\\
        0 & x \le 0
    \end{cases}
    $$

    The [LeakyReLU function](https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html):

    $$
    f(x) = \begin{cases}
        x & x \ge 0 \\\\
        x * \alpha & x < 0
    \end{cases}
    $$

    The [ELU function](https://docs.pytorch.org/docs/stable/generated/torch.nn.ELU.html):

    $$
    f(x) = \begin{cases}
        \alpha * (\exp(x) - 1) & x \le 0 \\\\
        x & x > 0
    \end{cases}
    $$

    The [softplus function](https://en.wikipedia.org/wiki/Softplus):

    $$
    f(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))
    $$

    The [Swish function](https://en.wikipedia.org/wiki/Swish_function):

    $$
    f(x) = x * \text{sigmoid}(\beta * x)
    $$

    The [GeLU function](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html):
    $$
    f(x) = x⋅Φ(x)
    $$
    """)
    return


@app.cell
def _(ActivationFunction, torch):
    # --- ReLU-based ---

    # ReLU
    class ReLU(ActivationFunction):
        def forward(self, x):
            return x * (x > 0).float()


    # LeakyReLU
    class LeakyReLU(ActivationFunction):
        def __init__(self, alpha=0.1):
            super().__init__()
            self.config["alpha"] = alpha

        def forward(self, x):
            return torch.where(x > 0, x, self.config["alpha"] * x)


    # ELU
    class ELU(ActivationFunction):
        def forward(self, x):
            return torch.where(x > 0, x, torch.exp(x) - 1)


    # Softplus
    class Softplus(ActivationFunction):
        def forward(self, x):
            return torch.log(1 + torch.exp(x))


    # Swish
    class Swish(ActivationFunction):
        def forward(self, x):
            return x * torch.sigmoid(x)


    # GELU
    class GELU(ActivationFunction):
        def forward(self, x):
            return (
                0.5
                * x
                * (
                    1
                    + torch.tanh(
                        torch.sqrt(torch.tensor(2.0 / torch.pi))
                        * (x + 0.044715 * torch.pow(x, 3))
                    )
                )
            )

    return ELU, GELU, LeakyReLU, ReLU, Softplus, Swish


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note: The GELU can be implemented in several ways:
    * the **exact version** is defined as $GELU(x) = x⋅Φ(x)$, where $Φ(x)$ is the **cumulative distribution function** of the **standard Gaussian distribution**.
    * the **approximation version** is defined as $GELU(x) = 0.5x(1 + tanh(√2/π(x + 0.044715x^3)))$, which is a **simplified form** of the exact version.

    Check [GELU - PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html).
    """)
    return


@app.cell
def _(
    ELU,
    GELU,
    Heaviside,
    LeakyReLU,
    ReLU,
    Sigmoid,
    Sine,
    Softplus,
    Swish,
    Tanh,
):
    act_fn_by_name = {
        "heaviside": Heaviside,
        "sine": Sine,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "relu": ReLU,
        "leakyrelu": LeakyReLU,
        "elu": ELU,
        "softplus": Softplus,
        "swish": Swish,
        "gelu": GELU,
    }
    return (act_fn_by_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing the Activation Functions
    """)
    return


@app.cell
def _(torch):
    def get_grads(act_fn, x):
        """
        Compute the gradients of an activation function at specified positions.

        Args:
            act_fn: The activation function
            x: The input tensor

        Returns:
            The gradients
        """
        # Ensure `x` is a leaf tensor with gradients enabled
        x = x.clone().requires_grad_(True)
        out = act_fn(x)

        # Check if the output actually depends on the input in a differentiable way
        if out.requires_grad:
            out.sum().backward()
            return x.grad if x.grad is not None else torch.zeros_like(x)
        else:
            # For non-differential functions, return zeros
            return torch.zeros_like(x)

    return (get_grads,)


@app.cell
def _(act_fn_by_name, get_grads, go, make_subplots, math, torch):
    def vis_act_fn(act_fn, fig, x, row, col, show_legend=False):
        # Run activation function
        y = act_fn(x)
        y_grads = get_grads(act_fn, x)

        # Push x, y and gradients back to cpu for plotting
        x_np = x.cpu().detach().numpy()
        y_np = y.cpu().detach().numpy()
        y_grads_np = y_grads.cpu().detach().numpy()

        # Add Activation Function trace
        fig.add_trace(
            go.Scatter(
                x=x_np,
                y=y_np,
                mode="lines",
                name="Activation",
                line=dict(color="blue", width=2),
                legendgroup="act",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )

        # Add Gradient trace
        fig.add_trace(
            go.Scatter(
                x=x_np,
                y=y_grads_np,
                mode="lines",
                name="Gradient",
                line=dict(color="orange", width=2, dash="dash"),
                legendgroup="grad",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )

        # set y-axis limits
        fig.update_yaxes(range=[-1.5, x_np.max() + 0.5], row=row, col=col)


    # Add activation functions
    act_fns = [act_fn() for act_fn in act_fn_by_name.values()]
    x_range = torch.linspace(-5, 5, 1000)

    # Layout setup
    cols = 2
    rows = math.ceil(len(act_fns) / cols)

    # Create subplots
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[fn.name for fn in act_fns],
        vertical_spacing=0.1,
    )

    for i, act_fn in enumerate(act_fns):
        r = (i // cols) + 1
        c = (i % cols) + 1
        # Only show legend for the very first plot to avoid duplicates
        vis_act_fn(act_fn, fig, x_range, r, c, show_legend=(i == 0))

    # Global layout adjustments
    fig.update_layout(
        height=rows * 250,
        width=800,
        title_text="Activation Functions and Their Gradients",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analyzing the effcet of activation functions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, building a neural network that views the images as 1D tensors and pushes them through a sequence of linear layers and a specific activation function.
    """)
    return


@app.cell
def _(nn):
    class BaseNetwork(nn.Module):
        def __init__(
            self,
            act_fn,
            input_size=784,
            num_classes=10,
            hidden_sizes=[512, 256, 256, 128],
        ):
            super().__init__()

            layers = []
            layer_sizes = [input_size] + hidden_sizes
            for layer_index in range(1, len(layer_sizes)):
                layers += [
                    nn.Linear(
                        layer_sizes[layer_index - 1], layer_sizes[layer_index]
                    ),
                    act_fn,
                ]
            layers += [nn.Linear(layer_sizes[-1], num_classes)]
            # `nn.Sequential` summarizes a list of modules into a single module, applying them in sequence
            self.layers = nn.Sequential(*layers)
            # We store all hyperparameters in a dictionary for saving and loading of the model
            self.config = {
                "act_fn": act_fn.config,
                "input_size": input_size,
                "num_classes": num_classes,
                "hidden_sizes": hidden_sizes,
            }

        def forward(self, x):
            # Reshape images to flat tensor
            x = x.view(x.size(0), -1)
            out = self.layers(x)
            return out

    return (BaseNetwork,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Also set up the helper functions for loading and storing the model weights.
    """)
    return


@app.cell
def _(BaseNetwork, act_fn_by_name, device, json, os, torch):
    def get_config_file(model_path, model_name):
        # Name of the file for storing hyperparameter details
        return os.path.join(model_path, model_name + ".config")


    def get_model_file(model_path, model_name):
        # Name of the file for storing network parameters
        return os.path.join(model_path, model_name + ".tar")


    def load_model(model_path, model_name, net=None):
        config_file, model_file = (
            get_config_file(model_path, model_name),
            get_model_file(model_path, model_name),
        )

        # Check if both files exist; if not, skip by returning None
        if not os.path.isfile(config_file) or not os.path.isfile(model_file):
            print(
                f"Warning: Files for {model_name} not found at {model_path}. Skipping..."
            )
            return None

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        if net is None:
            act_fn_name = config_dict["act_fn"].pop("name").lower()
            act_fn = act_fn_by_name[act_fn_name](**config_dict.pop("act_fn"))
            net = BaseNetwork(act_fn=act_fn, **config_dict)

        net.load_state_dict(torch.load(model_file, map_location=device))
        return net


    def save_model(model, model_path, model_name):
        config_dict = model.config
        os.makedirs(model_path, exist_ok=True)
        config_file, model_file = (
            get_config_file(model_path, model_name),
            get_model_file(model_path, model_name),
        )
        with open(config_file, "w") as f:
            json.dump(config_dict, f)
        torch.save(model.state_dict(), model_file)

    return get_model_file, load_model, save_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We would use the FashionMNIST dataset and `torchvision` to handle the data loading and preprocessing.

    `torchvision` provides popular datasets, model architectures, and preprocessing transforms for computer vision tasks.
    """)
    return


@app.cell
def _(DATASET_PATH, data, torch):
    import torchvision
    from torchvision.datasets import FashionMNIST
    from torchvision import transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = FashionMNIST(
        root=DATASET_PATH, train=True, transform=transform, download=True
    )
    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [50000, 10000]
    )
    test_set = FashionMNIST(
        root=DATASET_PATH, train=False, transform=transform, download=True
    )

    train_loader = data.DataLoader(
        train_set, batch_size=1024, shuffle=True, drop_last=False
    )
    val_loader = data.DataLoader(
        val_set, batch_size=1024, shuffle=False, drop_last=False
    )
    test_loader = data.DataLoader(
        test_set, batch_size=1024, shuffle=False, drop_last=False
    )
    return torchvision, train_loader, train_set, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note: PyTorch image grids are usually in the format `(Channels, Height, Width)`, but Plotly (and matplotlib) requires `(Height, Width, Channels)`.

    So remember to [`.permute()`](https://docs.pytorch.org/docs/stable/generated/torch.permute.html) the dimensions before plotting.
    """)
    return


@app.cell
def _(px, torch, torchvision, train_set):
    exmp_imgs = [train_set[i][0] for i in range(16)]

    img_grid = torchvision.utils.make_grid(
        torch.stack(exmp_imgs, dim=0), nrow=4, normalize=True, pad_value=0.5
    )
    img_grid = img_grid.permute(1, 2, 0)

    # Create the figure
    fashion_fig = px.imshow(img_grid, title="FashionMNIST examples")

    # Format
    fashion_fig.update_layout(
        width=600,
        height=600,
        xaxis_visible=False,
        yaxis_visible=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    fashion_fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing the gradient flow
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One important aspect of activation functions is how they **propagate the gradients** through the network. For a deep neural network:
    * If the gradient is considerably **smaller than 1**, the gradients will vanish as they are multiplied layer-by-layer.
    * If the gradient is **larger than 1**, the gradients will exponentially increase and might **explode**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note: plotly doesn't support Seaborn's `kde=True`(histogram + smooth curve). To mimics this, we can use `scipy.gaussian_kde` to calculate and overlay it as a `go.Scatter` trace.
    """)
    return


@app.cell
def _(F, data, device, gaussian_kde, go, make_subplots, np, train_set):
    def visualize_gradients(net, color_hex):
        """
        Inputs:
            net - Object of the `BaseNetwork`
            color - Color to visualize the histogram
        """
        # Set the model to eval mode
        net.eval()
        small_loader = data.DataLoader(train_set, batch_size=256, shuffle=False)
        imgs, labels = next(iter(small_loader))
        imgs, labels = imgs.to(device), labels.to(device)

        # Pass one batch through the network, and calculate the gradients for the weights
        net.zero_grad()
        preds = net(imgs)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots
        grads = {}
        for name, params in net.named_parameters():
            if "weight" in name:
                if params.grad is not None:
                    grads[name] = params.grad.data.view(-1).cpu().clone().numpy()
                else:
                    grads[name] = params.data.view(-1).cpu().clone().numpy() * 0
        net.zero_grad()

        # Plotting
        cols = len(grads)
        # Create subplot
        subplot_titles = [f"Layer: {k}" for k in grads.keys()]

        fig = make_subplots(
            rows=1,
            cols=cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
        )

        for i, (name, grad_values) in enumerate(grads.items()):
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=grad_values,
                    nbinsx=30,
                    marker_color=color_hex,
                    name=name,
                    opacity=0.6,
                    histnorm="probability density",
                ),
                row=1,
                col=i + 1,
            )

            # Add KDE Curve
            if len(grad_values) > 1 and np.var(grad_values) > 1e-10:
                kde = gaussian_kde(grad_values)
                x_curve = np.linspace(grad_values.min(), grad_values.max(), 100)
                y_curve = kde(x_curve)

                fig.add_trace(
                    go.Scatter(
                        x=x_curve,
                        y=y_curve,
                        mode="lines",
                        line=dict(color="black", width=1.5),
                        name=f"{name} KDE",
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=i + 1,
                )

            # Update axes per subplot
            fig.update_xaxes(title_text="Grad magnitude", row=1, col=i + 1)

        # Global layout
        fig.update_layout(
            title_text=f"Gradient distribution: {net.config['act_fn']['name']}",
            height=300,
            width=250 * cols,
            showlegend=False,
            margin=dict(t=60, b=40, l=20, r=20),
        )

        return fig

    return (visualize_gradients,)


@app.cell
def _(
    BaseNetwork,
    act_fn_by_name,
    device,
    mo,
    px,
    set_seed,
    visualize_gradients,
):
    figs = []
    colors = px.colors.qualitative.Plotly

    for idx, act_fn_name in enumerate(act_fn_by_name):
        set_seed(42)
        act_fn_for_grad = act_fn_by_name[act_fn_name]()
        net_actfn = BaseNetwork(act_fn=act_fn_for_grad).to(device)

        gradient_fig = visualize_gradients(
            net_actfn, color_hex=colors[idx % len(colors)]
        )
        figs.append(gradient_fig)

    mo.vstack(figs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For `sigmoid`, while the gradients for the output layer are very large with up to `0.1`, the input layer has the lowest gradient norm across all activation functions, with only `1e-5`. This is due to its small maximum gradient of `1/4`, and finding a suitable learning rate across all layers is challenging.

    The ReLU has a spike around `0`, which is caused by its zero-part on the left and dead neurons.

    The **initialization** of weight parameters is also crucial. By default, PyTorch uses the [Kaiming initialization](https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_) for **linear layers** optimized for ReLU activations.
    """)
    return


@app.cell(hide_code=True)
def _(
    CHECKPOINT_PATH,
    data,
    device,
    get_model_file,
    go,
    nn,
    optim,
    os,
    save_model,
    torch,
    tqdm,
    train_set,
    val_loader,
):
    def train_model(
        net, model_name, max_epoches=50, patience=7, batch_size=256, overwrite=False
    ):
        file_exists = os.path.isfile(get_model_file(CHECKPOINT_PATH, model_name))
        if file_exists and not overwrite:
            print("Model file already exists. Skipping training...")
        else:
            if file_exists:
                print("Model file exists, but will be overwritten")

            optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
            loss_module = nn.CrossEntropyLoss()
            train_loader_local = data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            )

            val_scores = []
            best_val_epoch = -1

            for epoch in range(max_epoches):
                # --- Training ---
                net.train()
                true_preds, count = 0.0, 0
                for imgs, labels in tqdm(
                    train_loader_local, desc=f"Epoch {epoch + 1}", leave=False
                ):
                    imgs, labels = imgs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    preds = net(imgs)
                    loss = loss_module(preds, labels)
                    loss.backward()
                    optimizer.step()
                    # Record statistics during training
                    true_preds += (preds.argmax(dim=-1) == labels).sum()
                    count += labels.shape[0]
                train_acc = true_preds / count

                # --- Validation ---
                val_acc = test_model(net, val_loader)
                val_scores.append(val_acc)
                print(
                    f"[Epoch {epoch + 1:2d}] Training accuracy: {train_acc * 100.0:05.2f}%, Validation accuracy: {val_acc * 100.0:05.2f}%"
                )

                if len(val_scores) == 1 or val_acc > val_scores[best_val_epoch]:
                    print("\t   (New best performance, saving model...)")
                    save_model(net, CHECKPOINT_PATH, model_name)
                    best_val_epoch = epoch
                elif best_val_epoch <= epoch - patience:
                    print(
                        f"Early stopping due to no improvement over the last {patience} epoches"
                    )
                    break

            epoches = list(range(1, len(val_scores) + 1))
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=epoches,
                    y=val_scores,
                    mode="lines+markers",
                    name="Validation Accuracy",
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=8),
                )
            )

            fig.update_layout(
                title=f"Validation performance of {model_name}",
                xaxis_title="Epoches",
                yaxis_title="Accuracy",
                yaxis=dict(tickformat=".2%"),
                template="plotly_white",
                height=400,
                width=700,
                margin=dict(l=40, r=40, t=60, b=40),
            )

            fig.show()


    def test_model(net, data_loader):
        net.eval()
        true_preds, count = 0.0, 0
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                preds = net(imgs).argmax(dim=-1)
                true_preds += (preds == labels).sum().item()
                count += labels.shape[0]
        test_acc = true_preds / count
        return test_acc

    return (train_model,)


@app.cell
def _(BaseNetwork, act_fn_by_name, device, set_seed, train_model):
    for act_fn_name_train in act_fn_by_name:
        print(f"Training BaseNetwork with {act_fn_name_train} activation ...")
        set_seed(42)
        act_fn_for_train = act_fn_by_name[act_fn_name_train]()
        net_actfn_train = BaseNetwork(act_fn=act_fn_for_train).to(device)
        train_model(
            net_actfn_train, f"FashionMNSIT_{act_fn_name_train}", overwrite=False
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model with the **sigmoid and heaviside activation fails**, which is not surprising.

    All other functions perform similarly. To get a more accurate conclusion, we might need to perform a **grid search** to train the models for multiple seeds and look at the averages.

    The **"optimal"** activation function **depends on many factors**, such as hidden size, number of layers, type of layers, dataset, optimizer, learning rate, etc.

    In general, **all the ReLU-like activation functions perform well**, with small gains for specific activation functions in specific networks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing the activation distribution
    """)
    return


@app.cell
def _(
    data,
    device,
    gaussian_kde,
    go,
    make_subplots,
    math,
    np,
    torch,
    train_set,
):
    def visualize_activations(net, color_hex="#1f77b4"):
        activations = {}

        net.eval()
        small_loader = data.DataLoader(train_set, batch_size=1024)
        imgs, labels = next(iter(small_loader))

        with torch.no_grad():
            imgs = imgs.to(device)
            imgs = imgs.view(imgs.size(0), -1)
            # Manually loop through layers to save activations
            for layer_index, layer in enumerate(net.layers[:-1]):
                imgs = layer(imgs)
                activations[layer_index] = imgs.view(-1).cpu().numpy()

        columns = 4
        rows = math.ceil(len(activations) / columns)

        subplot_titles = [
            f"Layer {k} - {net.layers[k].__class__.__name__}"
            for k in activations.keys()
        ]

        fig = make_subplots(
            rows=rows,
            cols=columns,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )

        for i, (layer_idx, vals) in enumerate(activations.items()):
            curr_row = (i // columns) + 1
            curr_col = (i % columns) + 1

            # 1. Add histogram
            fig.add_trace(
                go.Histogram(
                    x=vals,
                    nbinsx=50,
                    marker_color=color_hex,
                    opacity=0.6,
                    histnorm="probability density",
                    name=f"Layer {layer_idx}",
                    showlegend=False,
                ),
                row=curr_row,
                col=curr_col,
            )

            # 2. Add KDE
            if len(vals) > 0 and np.var(vals) > 1e-10:
                kde_sample = np.random.choice(
                    vals, min(len(vals), 1000), replace=False
                )
                kde = gaussian_kde(kde_sample)
                x_range = np.linspace(vals.min(), vals.max(), 100)
                y_kde = kde(x_range)

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_kde,
                        mode="lines",
                        line=dict(color="black", width=1),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=curr_row,
                    col=curr_col,
                )

        fig.update_layout(
            title_text=f"Activation distribution for activation function: {net.config['act_fn']['name']}",
            height=250 * rows,
            width=250 * columns,
            template="plotly_white",
            margin=dict(t=80, b=40, l=40, r=20),
        )

        return fig

    return (visualize_activations,)


@app.cell
def _(
    CHECKPOINT_PATH,
    act_fn_by_name,
    device,
    load_model,
    mo,
    os,
    px,
    visualize_activations,
):
    figs_activation = []
    plotly_colors = px.colors.qualitative.Plotly

    print(f"Checking for models in: {os.path.abspath(CHECKPOINT_PATH)}")

    for i_act, act_fn_name_act in enumerate(act_fn_by_name):
        model_name = f"FashionMNIST_{act_fn_name_act}"

        net_actfn_act = load_model(
            model_path=CHECKPOINT_PATH, model_name=model_name
        )

        if net_actfn_act is None:
            # Fallback for models trained with the old typo "FashionMNSIT"
            model_name_fallback = f"FashionMNSIT_{act_fn_name_act}"
            net_actfn_act = load_model(
                model_path=CHECKPOINT_PATH, model_name=model_name_fallback
            )
            if net_actfn_act is not None:
                model_name = model_name_fallback

        if net_actfn_act is None:
            print(
                f"Warning: Could not find model for {act_fn_name_act}. Skipping..."
            )
            continue

        print(f"✅ Found and loading: {model_name}...")

        try:
            net_actfn_act = net_actfn_act.to(device)

            act_fig = visualize_activations(
                net_actfn_act, color_hex=plotly_colors[i_act % len(plotly_colors)]
            )
            figs_activation.append(act_fig)
            print(f"   📊 Created plot for {act_fn_name_act}")
        except Exception as e:
            print(f"   ❌ Error visualizing {act_fn_name_act}: {e}")

    if figs_activation:
        print(f"Total plots created: {len(figs_activation)}. Rendering now...")
        display_output = mo.vstack(figs_activation)
    else:
        print("No models were loaded.")

    display_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As the model with sigmoid activation was not able to train properly, the activations are also **less informative** and **all gathered around `0.5`**(the activation value for `x=0`).

    The tanh shows a more **diverse** behavior. While for the **input layer** experience a larger amount of neurons to be **close to `-1` and `1`**, where the **gradients are close to zero**, the activations in the two consecutive layers are closer to zero. This is probably because the **input layers look for specific features** in the input image, and the consecutive layers combine those together. The activations for the **last layer** are again more **biased** to the **extreme points** because the classification layer can be seen as a weighted average of those values (the gradients push the activations to those extremes).

    The ReLU has a strong peak at `0`. The effect of **having no gradients for negative values** is that the network **does not have a Gaussian-like distribution after the linear layers**, but **a longer tail towards the positive values**. The LeakyReLU shows a very similar behavior, while ELU follows a more Gaussian-like distribution. The Swish activation seems to lie in between, although it is worth noting that Swish uses significantly higher values than other activation functions (up to `20`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Finding dead neurons in ReLU
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One known **drawback** of the ReLU activation is the occurrence of **“dead neurons”**, i.e., **neurons with no gradient for any training input**.

    The issue of dead neurons is that, as no gradient is provided for the layer, we cannot train the parameters of this neuron in the previous layer to obtain output values besides zero. For dead neurons to happen, the output value of a specific neuron of the linear layer before the ReLU has to be **negative** for all input images. Considering the large number of neurons we have in a neural network, it is not unlikely for this to happen.
    """)
    return


@app.cell
def _(ActivationFunction, device, nn, torch, tqdm, train_loader):
    def measure_number_dead_neurons(net):
        neurons_dead = [
            torch.ones(layer.weight.shape[0], device=device, dtype=torch.bool)
            for layer in net.layers[:-1] if isinstance(layer, nn.Linear)
        ]

        net.eval()
        with torch.no_grad():
            for imgs, labels in tqdm(train_loader, leave=False):
                layer_index = 0
                imgs = imgs.to(device)
                imgs = imgs.view(imgs.size(0), -1)
                for layer in net.layers[:-1]:
                    imgs = layer(imgs)
                    if isinstance(layer, ActivationFunction):
                        neurons_dead[layer_index] = torch.logical_and(
                            neurons_dead[layer_index],
                            (imgs == 0).all(dim=0)
                        )
                        layer_index += 1
        number_neurons_dead = [t.sum().item() for t in neurons_dead]
        print("Number of dead neurons:", number_neurons_dead)
        print("In percentage:", ", ".join([f"{(100.0 * num_dead / tens.shape[0]):4.2f}%" for tens, num_dead in zip(neurons_dead, number_neurons_dead)]))

    return (measure_number_dead_neurons,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dead neurons in an untrained network:
    """)
    return


@app.cell
def _(BaseNetwork, ReLU, device, measure_number_dead_neurons, set_seed):
    set_seed(42)
    net_relu = BaseNetwork(act_fn=ReLU()).to(device)
    measure_number_dead_neurons(net_relu)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dead neuron in a trained network:
    """)
    return


@app.cell
def _(CHECKPOINT_PATH, device, load_model, measure_number_dead_neurons):
    net_relu_trained = load_model(model_path=CHECKPOINT_PATH, model_name="FashionMNIST_relu").to(device)
    measure_number_dead_neurons(net_relu_trained)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The number of dead neurons indeed **decreased** in the later layers. However, it should be noted that **dead neurons are especially problematic in the input layer**. As the input does not change over epochs (the training set is kept as it is), training the network cannot turn those neurons back active. Still, the input data has usually a sufficiently high standard deviation to reduce the risk of dead neurons.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a deep network:
    """)
    return


@app.cell
def _(BaseNetwork, ReLU, device, measure_number_dead_neurons, set_seed):
    set_seed(42)
    net_relu_deep = BaseNetwork(act_fn=ReLU(), hidden_sizes=[256, 256, 256, 256, 256, 128, 128, 128, 128, 128]).to(device)
    measure_number_dead_neurons(net_relu_deep)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The number of dead neurons is significantly higher than before, which harms the gradient flow, especially in the first iterations. Hence, it is **advisable** to use **other nonlinearities** like **Swish** for **very deep networks**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conclusion
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * **Sigmoid** tends to **fail** deep neural networks as the highest gradient it provides is `0.25`, leading to vanishing gradients in early layers.
    * All **ReLU-based** activation functions have shown to **perform well**, and **besides** the original **ReLU**, **do not have the issue of dead neurons**.
    * When implementing your own neural network, it is **recommended** to **start with a ReLU-based** network and select the specific activation function based on the properties of the network.
    """)
    return


if __name__ == "__main__":
    app.run()
