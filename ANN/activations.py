import numpy as np 
import math


"""________________________________________ A C T I V A T I O N   F U N C T I O N S ______________________________________________
    Activation functions
    relu    -  Generally better than sigmoid but has a problem of losing or killing neurons
    sigmoid -  Was the commonly used
    softmax -  Better for mutli classification and it is stabilized by normalizing the values to avoid nan
    _______________________________________________________________________________________________________________________________
"""
def sigmoid(Z):
    activation = 1.0/(1.0+np.exp(-Z))
    return activation, Z

def sigmoid_plain(Z):
    activation = 1.0/(1.0+np.exp(-Z))
    return activation

def relu(Z):
    #return the maximum between two vectors
    activation = np.maximum(Z, 0)
    return activation, Z

def softmax(Z):
    #axis=0 sum along rows
    exp = np.exp(Z - np.max(Z))
    activation = exp/(np.sum(exp, axis=0))
    return activation, Z

def euler(Z):
    activation = np.exp(Z)
    return activation, Z

