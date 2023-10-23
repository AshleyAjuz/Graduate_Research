import csv
import os

#Function reads in a csv file and returns column names and data in separate lists
#CSV file format is assumed to be that odd ros contains the column name 
#and even rows contain the values delimited by commas

def read_csv(csv_fileName):
    
    #Create the lists to store label names and load profile values
    labels = []
    loadProfiles = []

    #Read from the csv file
    with open(csv_fileName) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line_count = 1
        for row in csv_reader:
            if line_count % 2 !=0:
                labels.append(str(row)[2:-2])
                line_count = line_count+1
            else:
                curList = (str(row)[2:-2]).split(",")
                loadProfiles.append([eval(i) for i in curList]) #Turn strings into integers
                line_count = line_count+1
    
    return( labels, loadProfiles)







            

