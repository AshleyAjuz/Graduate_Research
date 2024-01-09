import os
import numpy as np
import pandas as pd
import statistics
import random
from Functions.RNN_Forecastor import RNN_forecastor
from Functions.attackFunctions import attackFunctions
from Functions.evaluateRNN import evaluateRNN
from Functions.preProcess import preProcess
from Functions.clusterData import clustData
from Functions.create_csv import create_csv
from Functions.RNNAlg import findOptimalParams
from Functions.RNNAlg import RNN
from itertools import chain 


#Load in the benign datset of energy readings for 10 customers over a 3 year period
useCols =[f"Meter 0{i}" for i in range(10)]
benign_data = pd.read_csv("London_SM_data_total.csv", usecols=useCols).to_numpy().transpose()

#Need to create a label vector to denote that the readings are benign (y=0)
y_labs_B = [0 for i in range(len(benign_data))]

csv_path = "./Output/"


#Create attack data using the benign datset
attack_data = attackFunctions(benign_data, 1, csv_path)

#Need to create a label vector to denote that the readings are malicious (y=1)
y_labs_M =  [1 for i in range(len(attack_data))]

#Combine the benign_data and attack_data together to create the complete dataset (referred to as X_c in reference)
X_c = benign_data.tolist() + attack_data

#Create new overall y_labs vector
y_labs = y_labs_B + y_labs_M


# Need to perform over-sampling and feature scaling to balance the data since there is a 1:6 ratio of benign data to malicious data
# and to ensure that all of te inputs are within similar range to each other 
# Having an unablance dataset might consequently cause the RNN to bias to the malicious class
# Therefore, we need to redefine X_c and y_labs so that the ratio between the datasets are more similar
X_c, y_labs = preProcess(X_c, y_labs)

##Shuffle the datasets to make sure training and testing lists will have an even amount of benign and malicious reports
zipped = list(zip(X_c, y_labs))
random.shuffle(zipped)
X_c, y_labs = zip(*zipped)

## Testing with no clusters
#Create training dataset
idx = round(0.8*len(X_c))
X_train = X_c[0:idx]
y_train = y_labs[0:idx]

#Create testing dataset
X_test = X_c[idx:]
y_test = y_labs[idx:]

# Initialize values for RNN alg
I = 10 # num of epochs
N = 428 # num of neurons for each layer
K = 0.2 # Fraction for validation split

#Perform RNN
DR_values, FA_values, HD_values = evaluateRNN(X_train, y_train, X_test, y_test, N, K, I)


print("finished")