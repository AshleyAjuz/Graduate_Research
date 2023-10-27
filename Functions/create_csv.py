import os
import pandas as pd 

#Function that will create a csv of data given a list
#It is assumed that the data will be 2D list and multiple csv files will be created
def create_csv(data, csv_path):

    #Boolean variable to ensure code ran succesfully (Could also use try catch blocks if needed)
    codeFinished = False

    for i in range(len(data)):
        #Transform data into a csv file
        df = pd.DataFrame(data[i])
        df.to_csv(csv_path + f"Group{i}.csv")

     #Set codeFinished to true
    codeFinished = True
    return(codeFinished)