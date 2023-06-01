import numpy as np

def sigmoid(Z):
    '''
    compute sigmoid non-linear
    
    Args:
    Z -- linear output
    
    Return:
    A -- linear activation output
    activation_cache -- cache for backward propagation
    '''
    
    A = 1 / (1 + np.exp(-Z))
    
    activation_cache = Z
    
    return A, activation_cache

def sigmoid_backward(dA, cache):
    '''
    compute the backward gradients for sigmoid function
    
    Args:
    dA -- gradient for A
    cache -- intermediate variable for Z
    
    Return:
    dZ -- gradient for Z
    '''
    
    Z = cache
    A = 1 / (1 + np.exp(-Z))
    dZ = dA * A * (1-A)
    
    return dZ
    

def relu(Z):
    '''
    compute relu non-linear
    
    Args:
    Z -- linear output
    
    Return:
    A -- linear activation output
    activation_cache -- cache for backward propagation
    '''
    
    A = np.maximum(0, Z)
        
    activation_cache = Z
    
    return A, activation_cache

def relu_backward(dA, cache):
    '''
    compute the backward gradients for relu function
    
    Args:
    dA -- gradient for A
    cache -- intermediate variable for Z
    
    Return:
    dZ -- gradient for Z
    '''

    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    return dZ