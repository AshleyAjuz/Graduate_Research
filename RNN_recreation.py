import os
import numpy as np
import pandas as pd
import random
#from Functions.RNN_Forecastor import RNN_forecastor
from Functions.attackFunctions import attackFunctions
from Functions.evalRNN import evalRNN
from Functions.preProcess import oversample
from Functions.preProcess import preProcess
from Functions.plotDataset import plotDataset
import matplotlib.pyplot as plt
from itertools import chain 


#Dataset of energy readings for 10 customers over a 3 year period
#useCols =[f"Meter 0{i}" for i in range(1,10)]
#benign_data = pd.read_csv("InputData/London_SM_data_total.csv", usecols=useCols).to_numpy().transpose()

# Load in LCL dataset of aggregated energy readings for 5000 customers over a 3 year period
'''
test_B = pd.read_csv("InputData/london_energy.csv")
csv_path = "./Output/"
preProcess(test_B, csv_path)
'''


benign_data = pd.read_csv("Output/Tidy_LCL_Data.csv").to_numpy().transpose()[1:,:]

length, width = np.shape(benign_data)

csv_path = "./Output/"

#Need to create a label vector to denote that the readings are benign (y=0)
y_labs_B = [0 for i in range(len(benign_data))]

#Create attack data using the benign datset
attack_data = attackFunctions(benign_data, 1, csv_path)

#Create an array to store attack Ids
attack_IDs = [1,2,3,4,5]

#Need to create a label vector to denote that the readings are malicious (y=1)
y_labs_M =  [1 for i in range(len(attack_data))]

#Extend the attack IDs array to include both the benign and attack dataset
attack_IDs = y_labs_B + (attack_IDs * length)

#Hashmap to store occurences of attacks
attack_occ = {'0':0, '1':0, '2':0 , '3':0, '4':0, '5':0}

#Combine the benign_data and attack_data together to create the complete dataset (referred to as X_c in reference)
X_c = benign_data.tolist() + attack_data

#Create new overall y_labs vector
y_labs = y_labs_B + y_labs_M


# Need to perform over-sampling and feature scaling to balance the data since there is a 1:5 ratio of benign data to malicious data
# and to ensure that all of te inputs are within similar range to each other 
# Having an unablance dataset might consequently cause the RNN to bias to the malicious class
# Therefore, we need to redefine X_c and y_labs so that the ratio between the datasets are more similar
X_c, y_labs = oversample(X_c, y_labs)


#Compute how many extra samples were created
extr = len(y_labs) - len(attack_IDs)

#Updata samples in attack_IDs to account for oversampling
newSmplIDs = [0 for i in range(extr)]
attack_IDs.extend(newSmplIDs)

##Shuffle the datasets to make sure training and testing lists will have an even amount of benign and malicious reports
zipped = list(zip(X_c, y_labs, attack_IDs))
random.shuffle(zipped)
X_c, y_labs, attackIDs = zip(*zipped)

## Testing with no clusters
#Create training dataset
idx = round(0.8*len(X_c))
X_train = X_c[0:idx]
y_train = y_labs[0:idx]

#Data analysis for training
#plotDataset(X_train, y_train,"Training", "./Output/","TrainingPlots")

#Create testing dataset
X_test = X_c[idx:]
y_test = y_labs[idx:]

#Data analysis for testing set
#plotDataset(X_test, y_test, "Testing","./Output/","TestPlots")

# Initialize values for RNN alg
I = 10 # num of epochs
B = 350 # Batch Size
N = 215 # num of neurons for each layer

#Perform RNN
res, DR_values, FA_values, HD_values, Acc = evalRNN(X_train, y_train, X_test, y_test, N, B, I)

#Print out returned values
print(f"The DR_value is {DR_values}. The FPR_value is {FA_values}. The HD_values is {HD_values}")

#Determine the number of times an attack is detected
count = idx
for i in range(len(res)):
    cur_attack = attackIDs[count]
    if res[i] == 1:
        attack_occ[f"{cur_attack}"] += 1
    count += 1

#Compute confusion matrix to store which attacks are detected the most
cf = [0]
for i in range(1,6):
    totalOcc = sum(np.array(attackIDs) == i) #TODO: might need to use y_test here instead of attackIDs
    instancesFound = attack_occ[f"{i}"] 
    cf.append(float(instancesFound/totalOcc))

print(f"Attack 1:{cf[1]},Attack 2:{cf[2]}, Attack 3:{cf[3]}, Attack 4:{cf[4]}, Attack 5:{cf[5]} ")