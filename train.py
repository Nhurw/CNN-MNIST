import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import pickle

(train_X, train_y), (test_X, test_y) = mnist.load_data() ##Load data

train_XX = []
for i in range(0, len(train_X)):
    train_XX.append(train_X[i].flatten())
train_XX = np.array(train_XX)
train_yy = []
train_yy.append(train_y)
train_yy = np.array(train_yy)
train_yy = train_yy.T
data = np.hstack((train_yy, train_XX))
##Altering data structrure so that it works with this code
data = np.array(data)
m, n = data.shape

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255. ##Normalising the data values as a range from 0 to 1

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

##Initialising the weights and biases using a normal distribution
def initialise_parameters():
    # He initialization for ReLU activation: sqrt(2/n_inputs)
    # Input layer: 784 inputs (28x28 pixels)
    # Hidden layers: 256 neurons each
    # Output layer: 10 neurons (digits 0-9)
    
    Weights1 = np.random.randn(256, 784) * np.sqrt(2./784)
    biases1 = np.zeros((256, 1))
    
    Weights2 = np.random.randn(256, 256) * np.sqrt(2./256)
    biases2 = np.zeros((256, 1))
    
    Weights3 = np.random.randn(10, 256) * np.sqrt(2./256)
    biases3 = np.zeros((10, 1))
    
    return Weights1, biases1, Weights2, biases2, Weights3, biases3

def rectified_linear(initial):
    treated = []
    for i in range(0, len(initial)):
        treated.append(np.maximum(initial[i], 0))
    return treated

def softmax(initial):
    treated = np.exp(initial) / sum(np.exp(initial))
    return treated
    
def forward_propogation(Weights1, biases1, Weights2, biases2, Weights3, biases3, X):
    initial1 = Weights1.dot(X) + biases1
    treated1 = rectified_linear(initial1)
    initial2 = Weights2.dot(treated1) + biases2
    treated2 = rectified_linear(initial2)
    initial3 = Weights3.dot(treated2) + biases3
    treated3 = softmax(initial3)
    return initial1, treated1, initial2, treated2, initial3, treated3

def ReLU_deriv(initial):
    return initial > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_propogation(initial1, treated1, initial2, treated2, initial3, treated3, weights1, weights2, weights3, image_input, image_label):
    one_hot_Y = one_hot(image_label)
    delta_initial3 = treated3 - one_hot_Y
    delta_weights3 = 1 / m * delta_initial3.dot(np.array(treated2).T)
    delta_biases3 = 1 / m * np.sum(delta_initial3)
    delta_initial2 = weights3.T.dot(delta_initial3) * ReLU_deriv(initial2)
    delta_weights2 = 1 / m * delta_initial2.dot(np.array(treated1).T)
    delta_biases2 = 1 / m * np.sum(delta_initial2)
    delta_initial1 = weights2.T.dot(delta_initial2) * ReLU_deriv(initial1)
    delta_weights1 = 1 / m * delta_initial1.dot(image_input.T)
    delta_biases1 = 1 / m * np.sum(delta_initial1)
    return delta_weights1, delta_biases1, delta_weights2, delta_biases2, delta_weights3, delta_biases3

def update_paramaters(weights1, biases1, weights2, biases2, weights3, biases3, delta_weights1, delta_biases1, delta_weights2, delta_biases2, delta_weights3, delta_biases3, learning_rate):
    weights1 = weights1 - learning_rate * delta_weights1
    biases1 = biases1 - learning_rate * delta_biases1    
    weights2 = weights2 - learning_rate * delta_weights2  
    biases2 = biases2 - learning_rate * delta_biases2    
    weights3 = weights3 - learning_rate * delta_weights3  
    biases3 = biases3 - learning_rate * delta_biases3   
    return weights1, biases1, weights2, biases2, weights3, biases3 

def get_predictions(treated3):
    return np.argmax(treated3, 0)

def get_accuracy(predictions, image_label):
    print(predictions, image_label)
    return np.sum(predictions == image_label) / image_label.size

def train(image_input, image_label, learning_rate, iterations):
    weights1, biases1, weights2, biases2, weights3, biases3 = initialise_parameters()
    for i in range(iterations):
        initial1, treated1, initial2, treated2, initial3, treated3 = forward_propogation(weights1, biases1, weights2, biases2, weights3, biases3, image_input)
        delta_weights1, delta_biases1, delta_weights2, delta_biases2, delta_weights3, delta_biases3 = backward_propogation(initial1, treated1, initial2, treated2, initial3, treated3, weights1, weights2, weights3, image_input, image_label)
        weights1, biases1, weights2, biases2, weights3, biases3 = update_paramaters(weights1, biases1, weights2, biases2, weights3, biases3, delta_weights1, delta_biases1, delta_weights2, delta_biases2, delta_weights3, delta_biases3, learning_rate)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(treated3)
            print(round((get_accuracy(predictions, image_label) * 100), 4), "%")
    return weights1, biases1, weights2, biases2, weights3, biases3

weights1, biases1, weights2, biases2, weights3, biases3 = train(X_train, Y_train, 0.10, 500)

export_information = [weights1, biases1, weights2, biases2, weights3, biases3, test_X, test_y]

with open('data.pickle', 'wb') as file:
    pickle.dump(export_information, file)
    
