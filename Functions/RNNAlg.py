import os
import csv
import numpy as np
import math

'''Function performs GRU RNN Algorithm
Parameters include:
    -Training_data:
    -Training labels: true label for each training vector represented 
                      by one-hot vector y(x) = 0 for honest and y(x) = 1 for malicious
    -I: Number of iterations
    -M: Number of Mini Batches
    -L: Optimal number of layers
    -A_H: Activation Function
    -A_O: Softmax Function
    -n: Learning rate for gradient descent
'''
def RNN(training_data, training_labels, I, M, L, A_H, A_O, n):
    #Initialize parameters that will be determined
    U = [] # Array of Weights
    W = [] # Array of Weights
    V = [] # Array of Weights
    b = [] # Bias vector
    O = [] # Array to store the output
    K = len(training_data) # Total number of training samples (Num rows in training_data)
    layers = [i for i in range(1,L)]

    #Initialize iteration count variable (starting at 1)
    i = 1

    #Initialize the mini-batch count variable (starting at 1)
    m = 1

    #Initial changing variables
    z = []
    r = []
    h = []
    s = []

    while i < I:
        while m < M:
            for x in training_data[m]:
                #Perform feed forward
                for l in layers:
                    for t in x:
                        z[l][t] = A_H(O[l-1][t]*U[l][0] + s[l][t-1]*W[l][0] + b[l][0]) #Determing Update Gate at layer l
                        r[l][t] = A_H(O[l-1][t]*U[l][1] + s[l][t-1]*W[l][1] + b[l][1]) #Determing the Reset Gate at layer l
                        h[l][t] = math.tanh(O[l-1][t]*U[l][2] + ( np.multiply(s[l][t-1],r[l]) )*W[l][2] + b[l][2])#Determine the Hidden State at layer l
                        s[l][t] = np.multiply( (1-z[1]),h[l] ) + np.multiply( z[l], s[l][t-1] )
                        O[l][t] = A_O(V[l]*s[l][t]+b[l][3])#Output of layer 1
                    #end
                #end
            #Perform Backpropogation
            C = (training_labels[x]*math.log(O[l])) + ((1-training_labels[x])*math.log(1-O[l])) #Binary Cross Entropy equation for each training example
            #There is also a torch function for this that I could use for this
            chg_U = U[l-1]-U[l]
            chg_V = V[l-1]-V[l]
            chg_W = W[l-1]-W[l]
            chg_b = b[l-1]-b[l]

            #end
        #Update Weights and bias vector
        U[l] = U[l] - (n/K) * sum(chg_U * C)
        V[l] = V[l] - (n/K) * sum(chg_V * C)
        W[l] = W[l] - (n/K) * sum(chg_W * C)
        b[l] = b[l] - (n/K) * sum(chg_b * C)

    #Return optimal parameters
    return(U, W, V, b, layers)

