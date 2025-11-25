# Pseudocode
# --- Convolution Operation ---
def conv_layer_spatial(input, kernel, bias):
    output = convolution(input, kernel) + bias
    return activation(output)


# --- Nested Loops for Convolution Operation ---
def conv_layer_compute(input, kernel, bias):
    # Loop 1: Process each image in the batch
    for image in range(batch_size):
        # Loop 2 & 3: Move across image spatially
        for y in range(height):
            for x in range(width):
                # Loop 4: Compute each output feature
                for out_channel in range(num_output_channels):
                    result = bias[out_channel]
                    # Loop 5 & 6: Move across kernel window
                    for ky in range(kernel_height):
                        for kx in range(kernel_width):
                            # Loop 7: Process each image feature
                            for in_channel in range(num_input_channels):
                                # Get input value from current window position
                                in_y = y + ky
                                in_x = x + kx
                                # Perform multiply-accumulate opeartion
                                result += (
                                    input[image, in_y, in_x, in_channel]
                                    * kernel[ky, kx, in_channel, out_channel]
                                )
                    # Store the result for this position
                    output[image, y, x, out_channel] = result
