import os
import pathlib
from Functions.placeMeters import placeMeters
from Functions.plotMeterOutput import plotMeterOutput
from Functions.readMeters import readMeters
from Functions.attackFunctions import attackFunctions
from Functions.read_csv import read_csv
from Functions.change_loadProfiles import change_loadProfiles
from Functions.clusterData import FindOptimalClusters
from Functions.clusterData import clustData
from Functions.create_csv import create_csv
import numpy as np


#Create the path for which the dss circuit is located
script_path = os.path.dirname(os.path.abspath(__file__)) 

dss_file = pathlib.Path(script_path).joinpath("Feeders", "37Bus")

#Place smart meters depending on desired configuration
closeProx = ["L34,","L33","L26","L6"]
spreadOut = ["L12","L5","L19","L21"]

dss = placeMeters(dss_file,closeProx)

#Creating load curves
EVData = read_csv("../../InputData/Weekday_Load_Profiles.csv") #Eventually make the csv parameter dynamic


#Create list of example LoadNames and change load profiles
loadNames = ["s729a","s744a", "s728", "s730c", "s732c", "s731b"]

dss = change_loadProfiles(dss,EVData,loadNames)

#Read smart meter data and store in an array
csv_path = "./Output/"
energyReadings = readMeters(dss, len(closeProx), csv_path)

#Plot data
plot = plotMeterOutput("Output/MeterData.csv", csv_path)

#Find optimal cluster amount and store plot in Output folder
clustersFound = FindOptimalClusters(energyReadings, len(energyReadings), csv_path)

#Cluster data given the optimal number of clusters
optimal_clusters = 2
clusters = clustData(energyReadings, optimal_clusters)

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


