import os
import numpy as np
import pandas as pd
import statistics
import random
from Functions.attackFunctions import attackFunctions
from Functions.evaluateRNN import evaluateRNN
from Functions.preProcess import preProcess
from Functions.clusterData import clustData
from Functions.clusterData import FindOptimalClusters
from Functions.computeTrustworthiness import computeTrustworthiness


#Load in the benign datset of energy readings for 10 customers over a 3 year period
useCols =[f"Meter 0{i}" for i in range(1,10)]
benign_data = pd.read_csv("InputData/London_SM_data_total.csv", usecols=useCols).to_numpy().transpose()

#Need to create a label vector to denote that the readings are benign (y=0)
y_labs_B = [0 for i in range(len(benign_data))]

csv_path = "./Output/"

#Find optimal amount of clusters for the dataset
FindOptimalClusters(benign_data, len(benign_data), csv_path)

#Cluster data given the optimal number of clusters (found using the plot results from FindOptimalClusters)
optimal_clusters = 3
clusters, labels, cluster_means = clustData(benign_data, y_labs_B, optimal_clusters)

# Initialize values for RNN alg and trustworthiness alg
I = 30 # num of epochs
N = 428 # num of neurons for each layer
K = 0.2 # Fraction for validation split
threshold_1 = 0.5 #Threshold for distance based trustworthiness
threshold_2 = round(24160*0.5) #Threshold for if more than 50% of the data is compromised

# Store the metrics from each RNN results
DR_values = [0 for i in range(len(clusters))]
FA_values = [0 for i in range(len(clusters))] 
HD_values = [0 for i in range(len(clusters))] 


for i in range(len(clusters)):
    #Initialize the X_train and y_train datset for current cluster
    dataset = clusters[i][:]
    y_lab = labels[i]

    #Create attack data using the benign datset
    attack_data = attackFunctions(np.array(dataset), 1, csv_path)

    #Need to create a label vector to denote that the readings are malicious (y=1)
    y_labs_M =  [1 for i in range(len(attack_data))]

    #Combine the benign_data and attack_data together to create the complete dataset (referred to as X_c in reference)
    X_c = dataset + attack_data

    #Create new overall y_labs vector
    y_labs_total = y_lab + y_labs_M

    # Need to perform over-sampling and feature scaling to balance the data since there is a 1:6 ratio of benign data to malicious data
    # and to ensure that all of te inputs are within similar range to each other 
    # Having an unablance dataset might consequently cause the RNN to bias to the malicious class
    # Therefore, we need to redefine X_c and y_labs so that the ratio between the datasets are more similar
    X_c, y_labs = preProcess(X_c, y_labs_total)

    #Shuffle the datasets to make sure training and testing lists will have an even amount of benign and malicious reports
    zipped = list(zip(X_c, y_labs_total))
    random.shuffle(zipped)
    X_c, y_labs_total = zip(*zipped)

    #Create training dataset
    idx = round(0.8*len(X_c))
    X_train = X_c[0:idx]
    y_train = y_labs_total[0:idx]

    #Create testing dataset
    X_test = X_c[idx:]
    y_test = y_labs_total[idx:]


    #Perform RNN
    DR_values[i], FA_values[i], HD_values[i] = evaluateRNN(X_train, y_train, X_test, y_test, N, K, I)

    #Compute Trustworthiness of current cluster using a distance based trust measure
    clustTrust = computeTrustworthiness(X_test, cluster_means[i], threshold_1, threshold_2)
    
#Take the average of the metrics across the clusters to get the overall values
DR_overall = statistics.mean(DR_values)
FA_overall = statistics.mean(FA_values)
HD_values = statistics.mean(HD_values)

print("finished")