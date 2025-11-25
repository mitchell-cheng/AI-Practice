# Pseudocode
# --- MLP Layer ---
def mlp_layer_matrix(X, W, b):
    """
    X: input matrix (batch_size * num_inputs)
    W: weight matrix (num_inputs * num_outputs)
    b: bias vector (num_outputs)
    H = activation(matmul(X, W) + b)
    """
    return H


# --- Nested Loops for MLP Layer ---
def mlp_layer_compute(X, W, b):
    # Process each sample in the batch
    for batch in range(batch_size):
        # Compute each output neuron
        for out in range(num_outputs):
            Z[batch, out] = b[out]
            for in_ in range(num_inputs):
                Z[batch, out] += X[batch, in_] * W[in_, out]

    H = activation(Z)
    return H
