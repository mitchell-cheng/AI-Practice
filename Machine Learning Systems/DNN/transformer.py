# Pseudocode
# --- Attention Layer Matrix ---
def attention_layer_matrix(Q, K, V):
    """
    Q: Query matrix (batch_size * seq_len * d_model)
    K: Key matrix (batch_size * seq_len * d_model)
    V: Value matrix (batch_size * seq_len * d_model)
    """
    # Compute attention score
    scores = matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
    # Normalize scores
    weights = softmax(scores)
    # Combine values
    output = matmul(weights, V)

    return output


# --- Core computational pattern ---
def attention_layer_compute(Q, K, V):
    # Initialize outputs
    scores = np.zeros((batch_size, seq_len, seq_len))
    outputs = np.zeros_like(V)

    # Loop 1: Process each sequence in the batch
    for b in range(batch_size):
        # Loop 2: Compute attention for each query
        for i in range(seq_len):
            # Loop 3: Compare with each key
            for j in range(seq_len):
                # Compute attention scores
                for d in range(d_model):
                    scores[b, i, j] += Q[b, i, d] * K[b, j, d]
                scores[b, i, j] /= sqrt(d_k)

        # Apply softmax
        for i in range(seq_len):
            scores[b, i] = softmax(scores[b, i])

        # Loop 4: Compute values
        for i in range(seq_len):
            for j in range(seq_len):
                for d in range(d_model):
                    outputs[b, i, d] += scores[b, i, j] * V[b, j, d]

    return outputs


# --- Self Attention ---
def self_attention_layer(X, W_Q, W_K, W_V, d_k):
    """
    x: input tensor (batch_size * seq_len * d_model)
    W_Q: Query weight matrix (d_model * d_k)
    W_K: Key weight matrix (d_model * d_k)
    W_V: Value weight matrix (d_model * d_k)
    d_k: dimension of key vectors
    """

    Q = matmul(X, W_Q)
    K = matmul(X, W_K)
    V = matmul(X, W_V)

    scores = matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
    attention_weights = softmax(scores, dim=-1)
    output = matmul(attention_weights, V)

    return output


def multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads, d_k):
    outputs = []
    for i in range(num_heads):
        head_output = self_attention_layer(X, W_Q[i], W_K[i], W_V[i], d_k)
        outputs.append(head_output)

    concat_output = torch.cat(outputs, dim=-1)
    final_output = matmul(concat_output, W_O)

    return final_output
