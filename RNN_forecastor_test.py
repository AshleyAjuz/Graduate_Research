import numpy as np
import pandas as pd
import random
from Functions.RNN_Forecastor import RNN_forecastor
from Functions.RNN_Forecastor import split_sequence
from Functions.RNN_Forecastor import isUnderAttack
from Functions.RNN_Forecastor import computeMetrics
from Functions.attackFunctions import attackFunctions
from Functions.preProcess import oversample
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

                            ############### Testing with no clusters ############

#Dataset of energy readings for 400 customers over 3 years
benign_data = pd.read_csv("Output/Tidy_LCL_Data_2Y.csv").to_numpy().transpose()[1:,:]

#Need to create a label vector to denote that the readings are benign (y=0)
y_labs_B = [0 for i in range(len(benign_data))]

#Store an array for customer IDs in the benign dataset
custIds_benign = [x for x in range(len(benign_data))]

csv_path = "./Output/"

length, width = np.shape(benign_data)


#Determine split for training and testing
idx = round(0.7*width)

# Initialize values for RNN Forecastor
tstart = 0
tend = idx # End of the training dataset
n_steps = 7 #This is the step size that the GRU will be given for it to then predict the next value


#Perform RNN
mdl, rmse = RNN_forecastor(benign_data.T, tstart, tend, n_steps, csv_path, "PredictionsVsReal_NoClusters.png")


#Create attack data using the benign datset
attack_data = attackFunctions(benign_data, 1, csv_path)

#Create an array to store attack Ids
attack_IDs = [1,2,3,4,5]

#Need to create a label vector to denote that the readings are malicious (y=1)
y_labs_M =  [1 for i in range(len(attack_data))]

#Create customer ID array for the attack dataset
custIDs_attack = custIds_benign * 5


#Combine both sets of customer ID arrays
custIDs = custIds_benign + custIDs_attack

#Extend the attack IDs array to include both the benign and attack dataset
attack_IDs = y_labs_B + (attack_IDs * length)

#Hashmap to store occurences of attacks
attack_occ = {'0':0, '1':0, '2':0 , '3':0, '4':0, '5':0}

#Combine the benign_data and attack_data together to create the complete dataset (referred to as X_c in reference)
X_c = benign_data.tolist() + attack_data

#Create new overall y_labs vector
y_labs = y_labs_B + y_labs_M

#X_c, y_labs = oversample(X_c, y_labs)

##Shuffle the datasets 
zipped = list(zip(X_c, y_labs, custIDs, attack_IDs))
random.shuffle(zipped)
X_c, y_labs, custIDs, attackIDs = zip(*zipped)


#Initialize needed variables
sc = MinMaxScaler(feature_range=(0,1))
thrs = round(0.3*width)
eps = rmse * 0.3
overall_res = []

for i in range(len(X_c)):
    honest_data = benign_data[custIDs[i],:]
    cur_data = np.array(X_c[i][:])
    cur_attack = attackIDs[i]

    honest_data = honest_data.reshape(-1, 1)
    training_set_scaled = sc.fit_transform(honest_data)

    honest_data_X, honest_data_y = split_sequence(training_set_scaled, n_steps)
    honest_data_X = np.asarray(honest_data_X).astype('float64')

    honest_data_X = honest_data_X.reshape(honest_data_X.shape[0],honest_data_X.shape[1],1)

    predictions = mdl.predict(honest_data_X)

    cur_data = cur_data.reshape(-1, 1)
    cur_data_scaled = sc.fit_transform(cur_data)

    cur_data_X, cur_data_y = split_sequence(cur_data_scaled, n_steps)

    cur_res = isUnderAttack(honest_data_y,cur_data_y,rmse, eps,thrs)

    if cur_res:
        overall_res.append(1)
        attack_occ[f"{cur_attack}"] += 1
    else:
        overall_res.append(0)


#Get accuracy of results by comparing the entries in the y_labels with the overall_results
Acc = sum(np.array(overall_res) == np.array(y_labs))/len(y_labs)

#Get remaining metrics form compute metrics function
DR, FPR, HD = computeMetrics(y_labs, overall_res)

#Compute confusion matrix to store which attacks are detected the most
cf = [0]
for i in range(1,6):
    totalOcc = sum(np.array(attack_IDs) == i)
    instancesFound = attack_occ[f"{i}"] 
    cf.append(float(instancesFound/totalOcc))


#Print out returned values
print(f"The overall Accuracy is {Acc*100}. The DR_value is {DR}. The FPR_value is {FPR}. The HD_values is {HD}")
print(f"Attack 1:{cf[1]},Attack 2:{cf[2]}, Attack 3:{cf[3]}, Attack 4:{cf[4]}, Attack 5:{cf[5]} ")
