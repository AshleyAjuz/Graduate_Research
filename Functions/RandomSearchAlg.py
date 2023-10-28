import numpy as np
from RNNAlg import RNN
import math

################ NOT COMPLETE/TESTED ################

'''Function performs Random Search Algorith for tuning hyperparamets of the RNNAlg
    Options for hyper parameters were listed as the best performing models in the referenced
    article. There are other options for the hyper parameters that were not included here
Parameters include:
    -Training_data:
    -Training labels: true label for each training vector represented 
                      by one-hot vector y(x) = 0 for honest and y(x) = 1 for malicious
    -Test_data:
    -Test_labels
    -I: Number of iterations
    -M: Number of Mini Batches
    -n: Learning rate for gradient descent
'''
def RandomSearch(training_data, training_labels, test_data, test_labels, I, M, n):
    #Initial Hyper Parameters
    L = [2,3,4] #uniform number that represents the number of layers
    N = [428,310,215] #uniform number that represents the number of neurons
    O = ["Adam", "SGD", "Adamax"] #uniform distribution of optimization algorithms 
    A_H = ["Sigmoid","Hard Sigmoid","Tanh"] #uniform distribution of activation functions for the hidden layers
    A_O = "Softmax" #uniform distribution of activation functions for the output layers

    '''Initialize lists to store evaluation metrics:
        -DR: Detection rate. Measures the % of correctly detected malicious attacks
        -FA: False Acceptance rate. Measures the % of the honest samples that are falsely identified as malicious
        -HD: Highest Difference. Measures the Difference between DR and FA
    '''
    DR = []
    FA = []
    HD = []

    #Initialize iteration count variable (starting at 0)
    i = 0

    while i < 3:
        U, W, V, b, l = RNN(training_data, training_labels, L[i], I, M, A_H[i], A_O,n)

        #Evaluate the model with test dataset
        #DR[i], FA[i], HD[i] = evalMdl

    #Get the index of max DR, max HD, and min FA
    max_DR = DR.index(max(DR))
    max_HD = HD.index(max(HD))
    min_FA = FA.index(min(FA))

    #Select the most commonly choosen idex
    metrics = [max_DR, max_HD, min_FA]
    ind = max(set(metrics), key=metrics.count)

    #Return Optimized Hyper Parameters
    return(L[ind], N[ind], O[ind], A_H[ind], A_O[ind])
    

