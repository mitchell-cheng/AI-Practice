import numpy as np 
import matplotlib.pyplot as plt
import h5py
import scipy.io
import sklearn
import sklearn.datasets

def sigmoid(x):
    '''
    sigmoid function
    
    Args:
    x -- scalar
    
    Returns:
    s -- sigmoid(x)
    '''
    s = 1 / (1 + np.exp(-x))
    return s

def relu(x):
    '''
    relu function
    
    Args:
    x -- scalar
    
    Returns:
    s -- relu(x)
    '''
    
    s = np.maximum(0, x)
    return s

def load_params_and_grads(seed=1):
    np.random.seed(seed)
    
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 3)
    b2 = np.random.randn(3, 1)

    dW1 = np.random.randn(2, 3)
    db1 = np.random.randn(2, 1)
    dW2 = np.random.randn(3, 3)
    db2 = np.random.randn(3, 1)
    
    return W1, b1, W2, b2, dW1, db1, dW2, db2

def initialize_parameters(layer_dims):
    '''
    initialize_parameters
    
    Args:
    layer_dims -- layer dimensions
    
    Returns:
    parameters -- dictionary contains parameters
    '''
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    
    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt(2 / layer_dims[i-1])
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
        
    return parameters

def compute_cost(Y_pred, Y):
    '''
    compute model cost
    
    Args:
    Y_pred -- model prediction
    Y -- ground-truth label
    
    Returns:
    cost -- model cost
    '''
    
    probs = -(Y * np.log(Y_pred) + (1-Y) * np.log(1-Y_pred))
    cost = np.sum(probs)
    return cost

def forward_propagation(X, parameters):
    '''
    model forward pass
    
    Args:
    X -- training data
    parameters -- weights and bias
    
    Returns:
    A3 -- forward pass out
    cache -- intermediate variable for back prop
    '''
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (A1, Z1, W1, b1, A2, Z2, W2, b2, A3, Z3, W3, b3)
    
    return A3, cache
    
def backward_propagation(X, Y, cache):
    '''
    model backward pass
    
    Args:
    X -- training data
    Y -- training label
    cache -- all variables in forward pass
    
    Returns:
    grads -- gradients
    '''
    
    m = X.shape[1]
    (A1, Z1, W1, b1, A2, Z2, W2, b2, A3, Z3, W3, b3) = cache
    
    dZ3 = (1/m) * (A3 - Y)
    dW3 = np.dot(dZ3, A2.T)
    db3 = np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = np.dot(dZ1, X.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {
        'dW1': dW1, 'db1': db1, 'dZ1': dZ1, 'dA1': dA1,
        'dW2': dW2, 'db2': db2, 'dZ2': dZ2, 'dA2': dA2,
        'dW3': dW3, 'db3': db3, 'dZ3': dZ3 
    }
    
    return grads


def predict(X, Y, parameters):
    '''
    model prediction
    
    Args:
    X -- data for predicting
    Y -- ground-truth label
    
    Returns:
    p -- predictions
    '''
    
    m = X.shape[1]
    p = np.zeros((1, m), dtype=np.int)
    
    A3, cache = forward_propagation(X, parameters)
    
    for i in range(A3.shape[1]):
        if A3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
            
    print(f'Acc: {str(np.mean((p[0,:] == Y[0,:])))}')
    
    return p
       
def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);
    
    return train_X, train_Y, test_X, test_Y

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
    
def predict_dec(parameters, X):
    '''
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    '''
    
    A3, cache = forward_propagation(X, parameters)
    predictions = (A3 > 0.5)
    return predictions

def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)

    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return train_X, train_Y