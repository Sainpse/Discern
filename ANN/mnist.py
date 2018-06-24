import pandas as pd
import scipy as sc
import os
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork



dataPath_Train = os.path.join("..", "Datasets","mnist","mnist_train.csv")
dataPath_Test  = os.path.join("..", "Datasets","mnist","mnist_test.csv")

"""" 
Network Parameters
The number of neurons and layers in a network is defined by a list layer_dims where the length encodes the number of layers
and its elements as number of neurons
"""""
layer_dims = np.array([784,50,10])


## Reading the csv into a dataframe  usually for cleaning and exploring the data 
# @--Better use machine learning studio--@
df_Train = pd.read_csv(dataPath_Train, header=None)
df_Test = pd.read_csv(dataPath_Test, header=None)

"""" 
 Pre-Processing
 Convert into numpy array representation when all is preprocessed
 Y = labels
 X = Features
 transposing the matrices to make examples columns instead of rows
"""""
Train = df_Train.as_matrix()
Test  = df_Test.as_matrix()

#Labels
Y_Train = Train[0:,0]    
Y_Test  = Test[0:,0]

#Features
X_Train = np.transpose(Train[:,1:])
X_Test  = np.transpose(Test[:,1:])


#Verify shapes
print("Preview of Labels Dimensions")
print(Y_Train)
print(np.shape(Y_Train))
print("Preview Features Dimensions")
print(X_Train)
print(np.shape(X_Train))


#Declaring a new neural network
neuralnetwork = NeuralNetwork(layer_dims)
print("Training ...")
#Y_encoded = neuralnetwork.one_hot_encoding(Y_Train)
#neuralnetwork.train(X_Train,Y_encoded,1000,0.001)

#Viewing sample images
sample =  X_Test[:,0].reshape(28,28)
sample2 = X_Test[:,1].reshape(28,28)
sample3 = X_Test[:,2].reshape(28,28)

samples = [sample,sample2,sample3]
#vectors = [X_Test[:,0],X_Test[:,1],X_Test[:,2]]

for x in range(3):
    print(Y_Test[x])
    print(neuralnetwork.predict(samples[x]))