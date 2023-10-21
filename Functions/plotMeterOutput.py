import os
import csv
import matplotlib.pyplot as plt

def plotMeterOutput(csv_fileName):
    #Boolean variable to ensure code ran succesfully (Could also use try catch blocks if needed)
    codeFinished = False

    #Create variable to store all of the read in outputs
    total_readings = []
    
    #Create variable to store the number of objects present (num rows in the csv)
    num_objects = 0

    #Read from CSV file
    with open(csv_fileName) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader: # each row is a list
            total_readings.append(row[1:])
            num_objects = num_objects + 1

    #Delete first row from total_readings (these are the header values from the csv file)
    total_readings = total_readings[1:]
    num_objects = num_objects - 1

    #List to store time intervals from 0-48 (AKA the length of the meter readings list)
    number_list = [i for i in range(len(total_readings[1]))]

    #Plot all of the graphs
    for i in range(num_objects):
        plt.scatter(x=number_list, y=total_readings[i])

    plt.xlabel("Time [Hour]")
    plt.ylabel("Accumulated Energy [kWh]")
    plt.show()

    #Set codeFinished to true
    codeFinished = True
    return(codeFinished)

#### Main ###
plot = plotMeterOutput("Output/MeterData.csv")
print(plot)
