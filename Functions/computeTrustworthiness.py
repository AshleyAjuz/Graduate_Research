import os
import numpy as np
import math

'''Function determines the trustworthiness of a cluster of meters
Parameters include:
    -X_test: The new incoming data from meters within a given cluster
    -cluster_means: the cluster means of the benign dataset
    -threshold_1: used to determine if there is an anomaly in given timestamp based on the error between the cluster mean 
                  and a sample from X_test 
    -threshold_1: used to determine if an attack occurred based on if the number of anomalies exceeds this threshold
'''
def computeTrustworthiness(X_test, cluster_means, threshold_1, threshold_2):
    #Initialize variables
    dist_trust = 1 #Stores the overall trust score. Reference value for begins at 1
    diff_dist = [] #Stores the occurrence of an anomaly in energy consumption for each timestamp within the X_test samples

    alpha = 1/samples # Weighting factor that represents the percentage of trust in a single X_test sample

    samples, timeStamps = np.shape(X_test)

    for i in range(samples):
        curSample = X_test[i]
        for y in range(timeStamps):
            diff = math.sqrt((cluster_means[y] - curSample[y])^2)
            if diff > threshold_1:
                diff_dist[y] = 1 # Anomaly detected
            else:
                diff_dist[y] = 0 # No anomaly detected
        
        if sum(diff) > threshold_2:
            dist_trust = dist_trust - alpha
        
    
    return(dist_trust)
    