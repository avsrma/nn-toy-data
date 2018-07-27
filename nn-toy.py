# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 23:04:34 2018

@author: iamav
"""

import numpy as np
#import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def derivative(x):
    return x*(1-x)


training_inputs = np.array([[0,0,1],
                           [0,1,1],
                           [1,0,1],
                           [1,1,1]]) #4X3

training_outputs = np.array([[0,0,1,1]]).T #4X1
  
#plt.matshow(np.hstack((training_inputs, training_outputs)))          
#plt.show()                 
#assign weights

np.random.seed(1)

#synaptic_weights = 2 * np.random.random([3,1]) - 1
syn0 = 2 * np.random.random([3,3]) -1
syn1 = 2 * np.random.random([3,1]) -1

print ('Synaptic weights:')
print ('Syn0:', syn0)
print ('syn1' , syn1)

#training

for i in xrange(6000):
    #forward propagation

    l0 = training_inputs
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    #outputs = sigmoid(np.dot(l0, synaptic_weights))
    
    l2_error = training_outputs - l2
    #l1_error = training_outputs - l1
    
    if (i % 10000) == 0:
        print('Error:' + str(np.mean(np.abs(l2_error))))
        
    #backpropagation
    l2_delta = l2_error * derivative(l2)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * derivative(l1)
    
    #l1_delta = l1_error * derivative(l1)
    
    #update weights
#    syn1 = l1.T.dot(l2_delta)
#    syn0 = l0.T.dot(l1_delta)
    syn1 += np.dot(l1.T, l2_delta)
    syn0 += np.dot(l0.T, l1_delta)

#    training_inputs = np.array([[1,0,1],
#                           [1,1,1],
#                           [1,0,1],
#                           [1,1,1]]) #4X3
#    l0 = training_inputs
#    l1 = sigmoid(np.dot(l0, syn0))
#    l2 = sigmoid(np.dot(l1, syn1))

print('Trained outputs:', l2)
print(np.round(l2))
