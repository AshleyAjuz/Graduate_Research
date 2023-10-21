import os
import pandas as pd   


#Function for reading the time series data from the ciruit's energy meters
# And storing results in a csv file

def readMeters(dss, num_meters, csv_path):

    #Boolean variable to ensure code ran succesfully 
    codeFinished = False

    #Set up the timestamp intervals to read from the meters
    dss.text("set number=96")
    dss.text("set stepsize=.5h")
    dss.text("set mode=daily")

    #Create variable to store the total number of data (inputs)
    total_number = 48

    #Create  list to store all of the meter readings
    total_energy_kw_list = []
    #List to store time intervals from 0-48
    number_list = list()

    #Create a list for each smart meter
    for i in range(num_meters):
        SMList = []
        total_energy_kw_list.append(SMList)

    for number in range(total_number):
        dss.solution_solve() #solve the circuit at each time step

        #Store the total amount of energy captured by each meter for each time series
        number_list.append(number)

        #Reset circuit to start reading at the first meter
        dss.meters_first()

        #Store the power captured by each meters
        for i in range(num_meters):
            curList = total_energy_kw_list[i]
            curList.append(dss.meters_register_values()[0])
            dss.meters_next()

        dss.text(f"set casename={number}")

    #Go to parent directory
    os.chdir('../../')

    
    doesExist = os.path.exists(csv_path)
    if not doesExist:
       # Create a new directory if path does not exist
        os.makedirs(csv_path)

   #The following line transposes the matrix so that the cols become the rows and vice versa if needed
    #df = pd.DataFrame(total_energy_kw_list).T

    #Transform data into a csv file
    df = pd.DataFrame(total_energy_kw_list)
    df.to_csv(csv_path + "MeterData.csv")

    #Set code finished to true
    codeFinished = True

    return(codeFinished)
