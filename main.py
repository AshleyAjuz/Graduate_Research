import os
import pathlib
from Functions.placeMeters import placeMeters
from Functions.readMeters import readMeters
from Functions.attackFunctions import attackFunctions
from Functions.read_csv import read_csv
from Functions.change_loadProfiles import change_loadProfiles
from Functions.plotMeterOutput import plotMeterOutput
import numpy as np


#Create the path for which the dss circuit is located
script_path = os.path.dirname(os.path.abspath(__file__)) 

dss_file = pathlib.Path(script_path).joinpath("Feeders", "37Bus")

#Place smart meters depending on desired configuration
closeProx = ["L12,","L13","L24","L25"]
spreadOut = ["L12","L5","L19","L21"]

dss = placeMeters(dss_file,closeProx)

#Creating load curves
data = read_csv("../../InputData/Weekday_Load_Profiles.csv") #Eventually make the csv parameter dynamic

#Create list of example LoadNames
loadNames = ["S720c","S724b", "S722b", "S722c", "S724b", "S725b"]

dss = change_loadProfiles(dss,data,loadNames)

#Read smart meter data 
csv_path = "./Output/"

dataReadIn = readMeters(dss, len(closeProx), csv_path)

plot = plotMeterOutput("Output/MeterData.csv")

#Create attack data

#attacks = attackFunctions("Output/MeterData.csv","./Output/")
#print("Finished programing")


