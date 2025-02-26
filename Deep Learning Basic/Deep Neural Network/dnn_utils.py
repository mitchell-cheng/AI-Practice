import numpy as np
import h5py
import matplotlib.pyplot as plt

def load_data():
    '''
    load dataset
    '''
    training_data = h5py.File('./datasets/train_catvnoncat.h5', 'r')
    test_data = h5py.File('./datasets/test_catvnoncat.h5', 'r')
    training_set_x_orig = np.array(training_data['train_set_x'][:])
    training_set_y_orig = np.array(training_data['train_set_y'][:])
    test_set_x_orig = np.array(test_data['test_set_x'][:])
    test_set_y_orig = np.array(test_data['test_set_y'][:])

    classes = np.array(test_data['list_classes'][:])
    training_set_y_orig = training_set_y_orig.reshape((1, training_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return training_set_x_orig, training_set_y_orig, test_set_x_orig, test_set_y_orig, classes

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

# init two-layer model parameters

def init_parameters(n_x, n_h, n_y):
    '''
    random init for weights, init with zeros for bias
    
    Args:
    n_x -- size of input layer
    n_h -- neurons of hidden layer
    n_y -- neurons of output layer
    
    Return:
    parameters -- a python dictionary contains model parameters
    '''
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    
    return parameters

# init l-layers model parameters

def init_l_layer_parameters(layer_dims):
    '''
    random init for weights, init with zeros for bias
    
    Args:
    layer_dims -- a list contains layer dimensions
    
    Return:
    parameters -- a python dictionary contains model parameters
    '''
    
    np.random.seed(1)
    
    parameters = {}
    
    for i in range(1, len(layer_dims)):
        parameters['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters['b'+str(i)] = np.zeros((layer_dims[i], 1))

    return parameters

# linear_forward

def linear_forward(A, W, b):
    '''
    perform linear forward propagation
    
    Args:
    A -- previous layer input
    W -- current layer weights
    b -- current layer bias
    
    Return:
    Z -- linear forward output 
    linear_cache -- intermediate variable for backward propagation
    '''
    
    Z = W.dot(A) + b
    linear_cache = (A, W, b)
    
    return Z, linear_cache

# linear_activation_forward

def linear_activation_forward(A_prev, W, b, activation):
    '''
    perform linear forward propagation with non-linear
    
    Args:
    A_prev -- previous layer input
    W -- current layer weights
    b -- current layer bais
    activation -- type of activation function
    
    Return:
    A -- linear_activation output
    linear_activation_cache -- intermediate variable for backward propagation
    '''
    
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)
        
    linear_activation_cache = (linear_cache, activation_cache)
    return A, linear_activation_cache

# l-layer linear_activation_forward

def l_layer_linear_activation_forward(X, parameters):
    '''
    perform l-pass forward propagation with non-linear
    
    Args:
    X -- original input
    parameters -- a dictionary contains parameters
    
    Return:
    A -- linear_activation output
    cache -- intermediate variable for backward propagation
    '''
    A = X 
    L = len(parameters) // 2
    cache = []
    for i in range(1, L):
        A_prev = A
        A, linear_activation_cache = linear_activation_forward(A_prev, parameters['W'+str(i)], parameters['b'+str(i)], 'relu')
        cache.append(linear_activation_cache)
        
    A_out, linear_activation_cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    cache.append(linear_activation_cache)
        
    return A_out, cache

def compute_cost(Y, Y_predict):
    '''
    compute model cost
    
    Args:
    Y -- ground-truth labels
    Y_predict -- model output
    
    Return:
    cost -- model cost
    '''
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(Y_predict) + (1-Y) * np.log(1 - Y_predict)) / m
    cost = np.squeeze(cost)
    return cost

# linear_backward

def linear_backward(dZ, linear_cache):
    '''
    perform linear backward propagation
    
    Args:
    dZ -- gradient for Z
    linear_cache -- intermediate variables for backward propagation
    
    Return:
    linear_grads -- gradients
    '''
    
    A, W, b = linear_cache
    m = A.shape[1]
    
    dW = (1/m) * np.dot(dZ, A.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ) 
    
    linear_grads = {
        'dW': dW,
        'db': db,
        'dA_prev': dA_prev
    }
    
    return linear_grads


# linear_activation_backward

def linear_activation_backward(dA, linear_activation_cache, activation):
    '''
    perform linear backward propagation with non-linear
    
    Args:
    dA -- gradient for A
    linear_activation_cache -- intermediate variables for backward propagation
    activation -- decide to pick relu or sigmoid
    
    Return:
    linear_activation_grads -- gradients
    '''
    (linear_cache, activation_cache) = linear_activation_cache
    
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        linear_activation_grads = linear_backward(dZ, linear_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        linear_activation_grads = linear_backward(dZ, linear_cache)
        
    return linear_activation_grads

# l-layer linear_activation_backward

def l_layer_linear_activation_backward(Y_predict, Y, cache):
    '''
    perform l-pass forward propagation with non-linear
    
    Args:
    Y_predict -- forward pass output 
    Y -- ground-truth label
    cache -- intermediate cache
    
    Return:
    grads -- A dictionary with the gradients
    '''
    
    grads = {}
    L = len(cache)
    Y = Y.reshape(Y_predict.shape)
    
    dloss = - (np.divide(Y, Y_predict) - np.divide(1-Y, 1-Y_predict))
    current_cache = cache[L-1] 
    linear_activation_grads = linear_activation_backward(dloss, current_cache, 'sigmoid')
    dW = linear_activation_grads['dW']
    db = linear_activation_grads['db']
    dA_prev = linear_activation_grads['dA_prev']
    grads['dW' + str(L)] = dW 
    grads['db' + str(L)] = db 
    grads['dA' + str(L-1)] = dA_prev 
    
    for i in reversed(range(L-1)):
        current_cache = cache[i]
        linear_activation_grads = linear_activation_backward(grads['dA' + str(i+1)], current_cache, 'relu')
        dW = linear_activation_grads['dW']
        db = linear_activation_grads['db']
        dA_prev = linear_activation_grads['dA_prev']
        grads['dW' + str(i+1)] = dW
        grads['db' + str(i+1)] = db
        grads['dA' + str(i)] = dA_prev

    return grads

def update_params(params, grads, learning_rate):
    '''
    update weights and bias
    
    Args:
    parameters -- weights and bias
    grads -- gradients for weights and bias
    learning_rate -- learning rate alpha
    
    Return:
    updated_parameters -- updated parameters
    '''
    
    L = len(params) // 2
    parameters = params.copy()
    for i in range(L):
        parameters['W' + str(i+1)] = parameters['W' + str(i+1)] - learning_rate * grads['dW' + str(i+1)]
        parameters['b' + str(i+1)] = parameters['b' + str(i+1)] - learning_rate * grads['db' + str(i+1)]
    
    return parameters


def predict(X, y, parameters):
    '''
    using the parameters found by neural network to perform forward pass
    
    Args:
    X -- training data
    y -- corresponding labels
    
    Return:
    parameters -- weights and bias found by neural network
    '''
    m = X.shape[1]
    p = np.zeros((1, m))
    
    probas, caches = l_layer_linear_activation_forward(X, parameters)
    
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
            
    print(f'Acc {str(np.sum((p == y)/m))}')
    
    return p
