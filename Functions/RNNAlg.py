import os
import csv
import numpy as np
import math

'''Function performs findOptimalParam
Parameters include:
    -Training_data:
    -Training labels: true label for each training vector represented 
                      by one-hot vector y(x) = 0 for honest and y(x) = 1 for malicious
    -I: Number of iterations
    -L: Optimal number of layers
    -n: Learning rate for gradient descent
    -K: number of folds used for cross validation
'''
def findOptimalParams(training_data, training_labels, I, L, n, K):
    #Initialize weight matrices with random initial values
    U = np.random.uniform(0.0, 1.0, size=(L+1,len(training_data[0]))).tolist() # Array of Weights 
    W = np.random.uniform(0.0, 1.0, size=(L+1,len(training_data[0]))).tolist() # Array of Weights
    V = np.random.uniform(0.0, 1.0, size=(L+1,len(training_data[0]))).tolist() # Array of Weights
    b = np.random.uniform(0.0, 1.0, size=(L+2,len(training_data[0]))).tolist() # Bias vector
    o = np.random.uniform(0.0, 1.0, size=(L+1,len(training_data[0]))).tolist() # Array of Weights 
    layers = [i for i in range(1,L+1)]

    #Initialize iteration count variable (starting at 1)
    i = 1

    #Initial changing variables
    z = np.ones((L,len(training_data[0]))).tolist()
    r = np.ones((L,len(training_data[0]))).tolist()
    h = np.ones((L,len(training_data[0]))).tolist()
    s = np.ones((L,len(training_data[0]))).tolist()

    while i < I:
        #Initialize the mini-batch count variable (starting at 1)
        m = 0
        while m < len(training_data):
            #Get current training example
            inpData = training_data[m]

            # initial row  of o needs to be the training input data
            o[0] = inpData.copy()
            #Perform feed forward
            for l in layers:
                for t in range(1,len(inpData)):
                    # @ = matrix multiplication
                    z[l][t] = sigmoidFunc(o[l-1][t] * U[l][t] + s[l][t-1]*W[l][t] + b[l][t]) #Determing Update Gate at layer l
                    r[l][t] = sigmoidFunc(o[l-1][t]*U[l+1][t] + s[l][t-1]*W[l+1][t] + b[l+1][t]) #Determing the Reset Gate at layer l
                    h[l][t] = math.tanh(o[l-1][t]*U[l+2][t] + ( np.multiply(s[l][t-1],r[l]) )*W[l+2][t] + b[l+2][t])#Determine the Hidden State at layer l
                    s[l][t] = np.multiply( (1-z[l]),h[l] ) + np.multiply( z[l], s[l][t-1] )
                    o[l][t] = softmaxFunc(V[l]*s[l][t]+b[l+3])#Output of layer 1
                #end
            #end
            
            #Perform Backpropogation
            
            #Binary Cross Entropy equation for current training example
            C_x = training_labels[m]*math.log(o[l]) + (1-training_labels[m])*math.log(o[l]) 
            
            chg_U = U[l-1]-U[l]
            chg_V = V[l-1]-V[l]
            chg_W = W[l-1]-W[l]
            chg_b = b[l-1]-b[l]

            #Increment m
            m = m + 1

        #end

        #Update Weights and bias vector
        U[l] = U[l] - (n/K) * sum(chg_U * C_x)
        V[l] = V[l] - (n/K) * sum(chg_V * C_x)
        W[l] = W[l] - (n/K) * sum(chg_W * C_x)
        b[l] = b[l] - (n/K) * sum(chg_b * C_x)

        #Increment i
        i = i + 1
    #end

    #Return optimal parameters
    return(U, W, V, b, layers)



def sigmoidFunc(x):
    return 1/(1 + np.exp(-x))

def softmaxFunc(x):
    return np.exp(x) / sum(np.exp(x))

#There is also a torch function for this that I could use for this

def computeCrossEntropy(training_data, training_labels, o_w):
    #Initialize variable for all cross entropy values C
    C = []

    #Initialize S (number of rows in the training data)
    S = len(training_data)

    #Perform loop to  compute cross entropy for all samples in the dataset
    for i in range(S):
        eqn = (-1/S)*(training_labels[i]*math.log(o_w) + (1-training_labels[i])*math.log(o_w))
        C.append(eqn)

    #Return the value that is the smallest
    return (min(C))


def RNN(training_data, training_labels, U, W, V, b, l, N, O):
    #Initialize variables
    DR = 0
    FA = 0
    HD = 0



    #Return metrics
    return (DR, FA, HD)



