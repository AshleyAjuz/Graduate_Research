import csv
import numpy as np
from random import seed
from random import randint
import pandas as pd 

#Create the manipulated data for each smart meter
def attackFunctions(input_csv_fileName, output_csv_fileName):

    #Boolean variable to ensure code ran succesfully (Could also use try catch blocks if needed)
    codeFinished = False

    #Create variable to store all of the meter outputs
    total_meter_readings = []

    #Create variable to store the number of meters present
    num_meters = 0

    #Read from CSV file
    with open(input_csv_fileName) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader: # each row is a list
            total_meter_readings.append(row[1:])
            num_meters = num_meters + 1

    #Delete first row from total_meter_readings (these are the header values from the csv file)
    total_meter_readings = total_meter_readings[1:]
    num_meters = num_meters - 1

    #Create lists for each attack
    A1 = []
    A2 = []
    A3 = []
    A4 = []
    A5 = []
    A6 = []

    #Conduct first attack (reduce energy readings by some fraction a)
    a = np.random.uniform(0.1, 0.8, 1)[0]

    for x in range(num_meters):
        res = [a * K for K in total_meter_readings[x]]
        A1.append( res )

    #Conduct second attack (reduce energy readings by a dynamic amount)
    B = np.random.uniform(0.1, 0.8, len(total_meter_readings[1])).tolist()

    for x in range(num_meters):
        res = [B[i] * total_meter_readings[x][i] for i in range(len(total_meter_readings[x]))]
        A2.append( res )

    #Conduct third attack (report energy reading of 0 during a certain interval)
    t_i = np.random.randint(0, 42, 1)[0]
    t_f = np.random.randint(8, 47, 1)[0]

    #Create list of 1's
    list_multiplier = [1 for i in range(len(total_meter_readings[1]))]
    #Set specified interval to 0
    list_multiplier[t_i:t_f] = [0 for x in range(t_f-t_i)]

    for x in range(num_meters):
        res = [list_multiplier[i] * total_meter_readings[x][i] for i in range(len(total_meter_readings[x]))]
        A3.append( res )

    #Conduct fourth attack (report the mean of the energy readings )
    for x in range(num_meters):
        mean = sum(total_meter_readings[x])/len(total_meter_readings[x])
        list_mean = [mean for i in range(len(total_meter_readings[1]))]
        A4.append( list_mean )
    
    #Conduct fifth attack (report the mean of the energy readings that has been reduced by a dynamic amount )
    B_2 = np.random.uniform(0.1, 0.8, len(total_meter_readings[1])).tolist()

    for x in range(num_meters):
        mean = sum(total_meter_readings[x])/len(total_meter_readings[x])
        res = [B_2[i] * mean for i in range(len(B_2))]
        A5.append( res )
    
    #Not sure how to conduct 6 without prices matrix
        #If I were to get price matrix, I would find order the list from lowest pice to the highest
        #Then use those index values (from the reorder) to determine what order to make the total_energy_readings

    #Concatenate all of the attack lists
    Overall_Attack = []

    for x in range(num_meters):
        res = A1[x] + A2[x] + A3[x] + A4[x] + A5[x]
        Overall_Attack.append(res)

    #Transform data into a csv file
    df = pd.DataFrame(Overall_Attack)
    df.to_csv(output_csv_fileName)

    codeFinished = True
    return(codeFinished)



