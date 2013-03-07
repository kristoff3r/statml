#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import pylab as pl
from itertools import product
from numpy import dot

class NeuralNetwork:
    def __init__(self, n):
        self.w1 = np.random.normal(size=(n,2))      # Weights from input to hidden layer
        self.w2 = np.random.normal(size=(1,n+1))    # Weights from hidden layer to output layer
        self.dw1 = np.zeros((n,2))    # Last change in given weight
        self.dw2 = np.zeros((1,n+1))  # Last change in given weight
        self.M = n + 1 # Amount of neurons in hidden layer (incl. bias node)
        self.D = 1 + 1 # Amount of input neurons
        self.K = 1     # Amount of output neurons
        self.learning_rate = 0.03
        self.momentum_rate = 0.1
        self.hidden_start = self.D   # Index of first hidden layer neuron
        self.output_start = self.hidden_start + self.M  # Index of first output neuron
        self.total = self.output_start + self.K # Total amount of neurons

    # Activation function
    def h(self, a):
        return a / (1. + abs(a))

    # Derivative of activation function
    def h_(self, a):
        return 1. / pow(1. + abs(a),2)

    # Run the neural network
    def run(self, x):
        x_d = np.append(x, 1)                                 # Add bias to input layer
        w_in = np.append(dot(self.w1,x_d), 1).reshape(-1,1)   # Add bias to hidden layer
        y = dot(self.w2, np.apply_along_axis(self.h,1, w_in)) # Calculate output
        return y

    # Get or update weight
    def w(self, i, j, value=None): # If value != None then we update the weight with value
        if i < self.hidden_start: # Weight to input layer
            return 0
        elif i < self.output_start-1 and j < self.hidden_start: # Weight to hidden layer
            if value is not None:
                dw = value*self.learning_rate + self.momentum_rate * self.dw1[i-2,j]
                self.w1[i-2, j] -= dw
                self.dw1[i-2, j] = dw
            else:
                return self.w1[i-2, j]
        elif i == self.output_start and j >= self.hidden_start and j < self.output_start: # Weight to output layer
            if value is not None:
                dw = value*self.learning_rate + self.momentum_rate * self.dw2[0,j-2]
                self.w2[0,j-2] -= dw
                self.dw2[0,j-2] = dw
            else:
                return self.w2[0,j-2]
        else: # No connection exists
            return 0

    def train(self, data):
        data = data.copy()
        np.random.shuffle(data)
        for step in range(10):
            for row in data:
                x = row[0]
                y = row[1]
                z = [1,x]
                a = [0,0]

                # Compute node values
                for i in range(2, self.total):
                    a.append(sum([z[j]*self.w(i,j) for j in range(i)]))
                    if i >= self.hidden_start and i < self.output_start:
                        z.append(self.h(a[i]))
                    else:
                        z.append(a[i])

                delta = np.zeros(self.total - self.hidden_start + 2)
                delta[-1] = z[-1] - y
                for i in range(self.total-2, self.hidden_start-1, -1):
                    delta[i] = self.h_(a[i]) * sum([self.w(k,i)*delta[k] for k in range(i+1, self.total)])


                # Update weights
                gradient = np.zeros((len(z),len(z)))
                for (i, j) in product(range(len(delta)), range(len(z))):
                   gradient[i,j] = delta[i]*z[j]
                   self.w(i,j,delta[i]*z[j])
        #        print np.linalg.norm(gradient)




# Question 1.2
#sincTrain = np.loadtxt('data/sincTrain25.dt',ndmin=2)
sincTrain = np.loadtxt('data/sincMoreTrain.dt',ndmin=2)
NN = NeuralNetwork(20)
NN.train(sincTrain)
#print NN.w1
#print NN.w2
for row in sincTrain:
    v = NN.run(np.array(row[0]))
    pl.plot(row[0],row[1],'go')
    pl.plot(row[0],v,'ro')

pl.show()
