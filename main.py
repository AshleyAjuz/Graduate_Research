import os
import py_dss_interface
import pathlib
from Functions.placeMeters import placeMeters
from Functions.readMeters import readMeters
from Functions.attackFunctions import attackFunctions
import numpy as np


#Create the path for which the dss circuit is located
script_path = os.path.dirname(os.path.abspath(__file__)) 

dss_file = pathlib.Path(script_path).joinpath("Feeders", "37Bus")

#Place smart meters depending on desired configuration
closeProx = ["L12,","L13","L24","L25"]
spreadOut = ["L12","L5","L19","L21"]

dss = placeMeters(dss_file,closeProx)

#Read smart meter data 

csv_path = "./Output/"

dataReadIn = readMeters(dss, len(closeProx), csv_path)

#Create attack data

attacks = attackFunctions("Output/MeterData.csv","./Output/")

print("Finished programing")


