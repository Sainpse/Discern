import pandas as pd
import scipy as sc
import os
import numpy as np
import matplotlib.pyplot as plt
import activations 
import pickle
import random

class ANN:

    layer_dims = []
    Y          = []
    parameters = {}
    num_layers = 0
    batch_size = 0
    # Batch Gradient Descent as default
    optimization = "BGD" 

    ###### Utility Variables #######
    accuracylist = []

    
    def __init__(self, layer_dims, batch_size = 25, optimization ="BGD"):
        self.optimization = optimization
        self.batch_size  = batch_size
        self.initialize_net(layer_dims)

        
    """______________________________________________I N I T I A L I Z A T I O N___________________________________________

        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)

                        where l = current layer, hence (l-1) = previous layer
        ____________________________________________________________________________________________________________________
    """
    def initialize_net(self, layer_dims):
        
        np.random.seed(1)
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)          
    
        # Starting at 1 because zero is an input layer
        # Weights initialized to random values 
        # bias values initialized to zero
        for l in range(1, self.num_layers):
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
            self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            #insuring dimensions of arrays are up correct
            assert(self.parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(self.parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
        return self.parameters


    """_____________________L I N E A R   P O R T I O N   O F   F O R W A R D P R O P A G A T I O N______________________________________________
        Implementing the linear part of a layer's forward propagation.

        Arguments:
        X -- activations from previous layer (or input data): shape = (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        _______________________________________________________________________________________________________________________________
    """
    def linear_forward(self,X, W, b):
        Z = np.dot(W,X) + b
        
        assert(Z.shape == (W.shape[0], X.shape[1]))
        cache = (X, W, b)
    
        return Z, cache


    """_______________________________ F O R W A R D    P R O P A G A T I O N ______________________________________________________
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid", "relu" or "softmax"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently
        ____________________________________________________________________________________________________________________________
    """
    def linear_activation_forward(self,A_prev, W, b, activation):
        
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache"
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = activations.sigmoid(Z)

        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = activations.relu(Z) 
            
        elif activation == "softmax":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = activations.softmax(Z)
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache


    """_____________________________ F O R W A R D   P R O P A G A T I O N   C O N T I N U E ________________________________________
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        _____________________________________________________________________________________________________________________________
    """
    def L_model_forward(self,X):
        
        caches = []
        A = X
        L = self.num_layers
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L-1):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev,self.parameters["W" + str(l)], self.parameters["b" + str(l)], "relu")
            caches.append(cache)
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache =  self.linear_activation_forward(A,self.parameters["W" + str(L-1)], self.parameters["b" + str(L-1)], "softmax")
        caches.append(cache)
        assert(AL.shape == (10,X.shape[1]))        
        return AL, caches

    """_________________________________________ C O S T    F U N C T I O N _____________________________________________________
        Implement the cost function defined by equation  D(AL,Y) = -sum(Y(LOG(AL))
        used when output is a proobability distribution e.i last layer is a softmax

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        __________________________________________________________________________________________________________________________
    """
    
    def compute_cost(self,AL, Y):
        
        m = Y.shape[1]

        # Compute loss from aL and y.
        cost = -1*np.sum(Y*np.log(AL))/m
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        return cost

    """____________________________ L I N E A R   P O R T I O N   O F  B A C K P R O P A G A T I O N _____________________________
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        __________________________________________________________________________________________________________________________
    """
    def linear_backward(self, dZ, cache):
    
        A_prev, W, b = cache
        m = A_prev.shape[1]


        dW = (1/m)*(np.dot(dZ,np.transpose(A_prev)))
        db = (1/m)*(np.sum(dZ, axis=1, keepdims=True))
        dA_prev = np.dot(np.transpose(W),dZ)
       
    
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        return dA_prev, dW, db

    def sigmoid_backward(self,dA, activation_cache):
        Z = activation_cache
        d = self.sigmoid_plain(Z)*(1 -self.sigmoid_plain(Z))
        dZ = np.multiply(dA,d) # Chain Rule
        return dZ

    def relu_backward(self,dA, activation_cache):
        Z = activation_cache
        Z[Z>0] = 1
        Z[Z<0] = 0.01
        dZ = np.multiply(dA,Z)   # Chain Rule
        return dZ

    def softmax_backward(self, dA, activation_cache):
        return dA - self.Y

    """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we stored for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    def linear_activation_backward(self, dA, cache, activation):
        
        linear_cache, activation_cache = cache
        
        if activation == "relu":
           
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
           
            
        elif activation == "sigmoid":
            
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "softmax":
            dZ = self.softmax_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
           
        
        return dA_prev, dW, db


        
    """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
    """
    def L_model_backward(self, AL, Y, caches):
        
        grads = {}
        L = len(caches) # the number of layers
        #m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        dAL = np.divide(Y,AL) #Derivative of the cost function D(Y,AL) = Y*LOG(AL)
        
       
        current_cache = caches[L-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(AL, current_cache, "softmax")
    
        
        for l in reversed(range(L-1)):
            
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+2)], current_cache, "relu")
            grads["dA" + str(l + 1)] =  dA_prev_temp
            grads["dW" + str(l + 1)] =  dW_temp
            grads["db" + str(l + 1)] =  db_temp
          

        return grads    

     
    """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                    parameters["W" + str(l)] = ... 
                    parameters["b" + str(l)] = ...
    """
    def update_parameters(self, grads, learning_rate):
       
        
        L = len(self.parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
       
        for l in range(L):
            W = self.parameters["W"  + str(l + 1)]
            self.parameters["W" + str(l + 1)] = W - learning_rate*grads["dW" + str(l + 1)]
            b = self.parameters["b" + str(l + 1)]
            self.parameters["b" + str(l + 1)] = b - learning_rate*grads["db" + str(l + 1)]



    """
        Training function that prints out the prediction accuracy at each cycle
    """
    def train(self, X,Y, cycles, learning_rate):

        X, self.Y = self.learning_optimization(X,Y)
        
        for cyc in range(cycles):
            ActivationLearn, caches = self.L_model_forward(X)
          
            errorRate = self.compute_cost(ActivationLearn,self.Y)
            accu = 100 - (errorRate*100)
            print("Accuracy: " + str(accu))
            print("cycle "+ str(cyc))

            self.accuracylist.append(accu)
           
            grads = self.L_model_backward(ActivationLearn,self.Y,caches)
            self.update_parameters(grads, learning_rate)

        ### Saving Object for later evaluation
        with open(self.optimization+".pkl", 'wb') as f:
            pickle.dump(self.accuracylist, f)

        print("Final Training Accuracy: " + str(100 -(errorRate*100)))


    """ encoding the digits the digits
    """  
    def one_hot_encoding(self,Y):

        classes = list(np.unique(Y))
        hot_encoding = np.zeros((len(classes), len(Y)))
        index = 0
        for x in Y:
            hot_encoding[classes.index(x)][index] = 1
            index += 1
        return hot_encoding


    """
        Prediction function returning the prediction probabilities of  digits between 0 - 9
    """
    def predict(self,Image):
        prediction, cache = self.L_model_forward(Image)
        return prediction


    """
        Learning Optimization
            - Mini-Batch
                Takes a subset of the training set randomly depending on the batchsize
                (currently it doesnt mind repeating observations)
    """
    def mini_batch(self,X, Y):
        minBatch = self.batch_size
        labels = []
        #Assigning a holder row
        batch = np.array((X[:,0]))
    
        for i in range(minBatch):
            i = random.randint(0,X.shape[1] - 1)
            #Corrosponding label
            labels.append(Y[:,i]) 
            # i randomly choosing an example
            batch = np.vstack((batch,X[:,i]))

        #remove the holder row 
        batch  = np.delete(batch,0,0)
        labels = np.array(labels)
        batch  = batch.T
        labels = labels.T
        assert(batch.shape == (X.shape[0], self.batch_size))
        assert(labels.shape == (Y.shape[0], self.batch_size))

        return batch, labels

    """
        Train the model with one random sample from the training data
    """
    def stochastic(self,X,Y):
        i = random.randint(0,X.shape[1] - 1)
        X = X[:,i] #random observation
        Y = Y[i]  #corrosponding label
        return X, Y

        
    """
        Learning optimization function for gradient descent regarding observation sampling
            BGD  - Batch Gradient Descent
            MBGD - Mini Batch Gradient Descent
            SGD  - Stochastic Gradient Descent
    """
    def learning_optimization(self, X, Y):
        #Batch Gradient Descent
        if self.optimization == "BGD":
            return X, Y
        elif self.optimization == "MBGD":
            return self.mini_batch(X, Y)
        else: # Stochastic Gradient descent
            return self.stochastic(X,Y)





