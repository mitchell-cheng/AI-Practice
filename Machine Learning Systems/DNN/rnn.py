# Pseudocode
# --- RNN Layer Step ---
def rnn_layer_step(x_t, h_prev, W_hh, W_xh, b):
    """
    x_t: input at time t (batch_size * input_dim)
    h_prev: previous hidden state (batch_size * hidden_dim)
    W_hh: recurrent weights (hidden_dim * hidden_dim)
    W_xh: input weights (input_dim * hidden_dim)
    b: bias vector (hidden_dim)
    """
    h_t = activation(matmul(h_prev, W_hh) + matmul(x_t, W_xh) + b)
    return h_t


# --- Recurrent Layer Computation ---
def rnn_layer_compute(x_t, h_prev, W_hh, W_xh, b):
    # Initialize next hidden state
    h_t = np.zeros_like(h_prev)

    # Loop 1: Process each sequence in the batch
    for batch in range(batch_size):
        # Loop 2: Compute recurrent contribution
        # (h_prev * W_hh)
        for i in range(hidden_dim):
            for j in range(hidden_dim):
                h_t[batch, i] += h_prev[batch, j] * W_hh[j, i]

        # Loop 3: Compute input contribution
        for i in range(hidden_dim):
            for j in range(input_dim):
                h_t[batch, i] += x_t[batch, j] * W_xh[j, i]

        # Loop 4: Add bias and apply activation
        for i in range(hidden_dim):
            h_t[batch, i] = activation(h_t[batch, i] + b[i])

    return h_t
