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


#Dataset of energy readings for 700 customers over 3 years
benign_data = pd.read_csv("Output/Tidy_LCL_Data_2Y.csv").to_numpy().transpose()[1:,:]

length, width = np.shape(benign_data)

#Store an array for customer IDs in the benign dataset
custIds_benign = [x for x in range(length)]

csv_path = "./Output/"

#Generate 10 random numbers that will be used to select 10 customers from the benign dataset
rndIdxs = np.random.random_integers(0, length, 10).tolist()


#Determine split for training and testing
idx = round(0.7*width)

# Initialize values for RNN Forecastor
tstart = 0
tend = idx # End of the training dataset
n_steps = 7 #This is the step size that the GRU will be given for it to then predict the next value

for customer in rndIdxs:

    #Perform RNN
    mdl, rmse = RNN_forecastor(benign_data[customer].T, tstart, tend, n_steps, csv_path, "PredictionsVsReal_NoClusters.png")

    #Create attack data using the benign datset
    attack_data = attackFunctions(benign_data[customer], 1, csv_path)


    #Need to create a label vector to denote that the readings are malicious (y=1)
    y_labs_M =  [1 for i in range(len(attack_data))]


    #Combine the benign_data and attack_data together to create the complete dataset (referred to as X_c in reference)
    X_c = attack_data
    X_c.append(benign_data[customer].tolist())

    #Create new overall y_labs vector
    y_labs = y_labs_M + [0]


    ##Shuffle the datasets 
    zipped = list(zip(X_c, y_labs))
    random.shuffle(zipped)
    X_c, y_labs = zip(*zipped)


    #Initialize needed variables
    sc = MinMaxScaler(feature_range=(0,1))
    thrs = n_steps
    eps = rmse * 0.3
    overall_res = []

    for i in range(len(X_c)):
        honest_data = benign_data[customer]
        cur_data = np.array(X_c[i][:])

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
        else:
            overall_res.append(0)


    #Get accuracy of results by comparing the entries in the y_labels with the overall_results
    Acc = sum(np.array(overall_res) == np.array(y_labs))/len(y_labs)

    #Get remaining metrics form compute metrics function
    DR, FPR, HD = computeMetrics(y_labs, overall_res)

    #Print out returned values
    print(f"Results for Customer {customer}: Accuracy = {Acc*100}. DR_value = {DR*100}. FPR_value = {FPR*100}. The HD_values = {HD*100}")

