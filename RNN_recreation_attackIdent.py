import os
import numpy as np
import pandas as pd
import random
from Functions.attackFunctions import attackFunctions
from Functions.evalRNN_attackId import evalRNN_attackId
from Functions.plotCustData import plotCustData
from Functions.preProcess import oversample
import matplotlib.pyplot as plt
from itertools import chain 


#Dataset of energy readings for aboug 700 customers over a 2 year period
benign_data = pd.read_csv("Output/Tidy_LCL_Data_2Y.csv").to_numpy().transpose()[1:,:]

length, width = np.shape(benign_data)

csv_path = "./Output/"

#Need to create a label vector to denote that the readings are benign (y=0)
y_labs_B = [0 for i in range(length)]

#Create attack data using the benign datset
attack_data = attackFunctions(benign_data, 1, csv_path)

#Need to create a label vector to denote that the readings are malicious (y=1)
y_labs_M =  [1 for i in range(len(attack_data))]

#Extend the attack IDs array to include both the benign and attack dataset
attack_IDs = y_labs_B.copy()
for i in range(1,6):
    curAttack = [i for x in range(length)]
    attack_IDs += curAttack


#Hashmap to store occurences of attacks
attack_occ = {'0':0, '1':0, '2':0 , '3':0, '4':0, '5':0}

#Combine the benign_data and attack_data together to create the complete dataset (referred to as X_c in reference)
X_c = benign_data.tolist() + attack_data

#Select random customer's data to plot
rndCust = np.random.randint(0, width, 1)[0]
plotCustData(rndCust,X_c[rndCust], attack_data, 100, csv_path)


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

#Create training dataset (using the attackIDs as the y labels)
idx = round(0.8*len(X_c))
X_train = X_c[0:idx]
y_train = attackIDs[0:idx]


#Create testing dataset
X_test = X_c[idx:]
y_test = attackIDs[idx:]

# Initialize values for RNN alg
I = 10 # num of epochs
B = 350 # Batch Size
N = 215 # num of neurons for each layer

#Perform RNN
evalRNN_attackId(X_train, y_train, X_test, y_test, N, B, I)


