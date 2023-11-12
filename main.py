import os
import pathlib
from Functions.placeMeters import placeMeters
from Functions.plotMeterOutputs import plotMeterOutputs
from Functions.readMeters import readMeters
from Functions.GetMeterOutput import GetMeterOutput
from Functions.attackFunctions import attackFunctions
from Functions.change_loadProfiles import change_loadProfiles
from Functions.clusterData import FindOptimalClusters
from Functions.clusterData import clustData
from Functions.create_csv import create_csv
import numpy as np
import pandas as pd
import py_dss_interface


#Create the path for which the dss circuit is located
script_path = os.path.dirname(os.path.abspath(__file__)) 

dss_file = pathlib.Path(script_path).joinpath("Feeders", "37Bus")


#Place smart meters depending on desired configuration
closeProx = ["l34,","l33","l26","l6","l4","l27","l15","l14","l17","l16"]
spreadOut = ["l12","l5","l19","l21"]

dss = placeMeters(dss_file,closeProx)


#Import typical household loadprofiles to assign to each load in the feeder
HouseholdLoadProfile = pd.read_csv("../../InputData/HouseholdLoadData.csv")
names_loads = dss.loads_all_names()

#Import EV loadprofiles
EVProfileData = pd.read_csv("../../InputData/DOE_EVData.csv").iloc[1,1:].tolist()
#Create list of loads to have EVS
MeteredLoads = ["s729a","s744a", "s728", "s730c", "s732c", "s731b"]

dss = change_loadProfiles(dss,HouseholdLoadProfile,names_loads, MeteredLoads, EVProfileData) 


#Read smart meter data and store in an array
csv_path = "./Output/"
voltageProfiles = GetMeterOutput(dss, csv_path)

#Plot the Meter Data
plotMeterOutputs(voltageProfiles, dss.meters_all_names(), csv_path)


#Find optimal cluster amount and store plot in Output folder
transposed_VoltProf = np.transpose(voltageProfiles)
clustersFound = FindOptimalClusters(transposed_VoltProf, 11, csv_path)

#Cluster data given the optimal number of clusters (found using the plot results from FindOptimalClusters)
optimal_clusters = 3
clusters = clustData(transposed_VoltProf, optimal_clusters)


#Create a collection of csv files from the clusters data to store data that are in the same clusters together
csvsCreated = create_csv(clusters,csv_path) 

'''
Create attack data for each of the csv files created for each cluster

for i in range (len(clusters)):
    input_csv_file = f"Output/Group{i}.csv"
    output_csv_file =f"./Output/Group{i}_Attack_Data.csv"
    attackFunctions(input_csv_file, output_csv_file)

'''

print("Finished programing")


