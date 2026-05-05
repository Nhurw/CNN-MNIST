import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import pickle

file_path = 'data.pickle'

with open(file_path, 'rb') as file:
    weights_and_biases = pickle.load(file)

W1 = weights_and_biases[0]
b1 = weights_and_biases[1]
b1 = b1.flatten()
W2 = weights_and_biases[2]
b2 = weights_and_biases[3]
b2 = b2.flatten()
W3 = weights_and_biases[4]
b3 = weights_and_biases[5]
b3 = b3.flatten()
test_X = weights_and_biases[6]
test_y = weights_and_biases[7]

test_XX = [] 
for i in range(0, len(test_X)):
    test_XX.append(test_X[i].flatten()) 

test_XX = np.array(test_XX)
test_XX = test_XX / 255
test_yy = np.array(test_y)

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def get_predictions(A3):
    return np.argmax(A3, 0)

def prediction(input):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, input)
    prediction = get_predictions(A3)
    return prediction

Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, test_XX[0])

for i in range(400, 450):
    fig = plt.figure
    plt.imshow(test_X[i], cmap='gray')
    plt.show()
    print(prediction(test_XX[i]))


