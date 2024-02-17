import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def preProcess(df, csv_path):
     #Get all of the customer IDs
    custIds = df["LCLid"].unique().tolist()

    #Get all unique Dates
    dates = df["Date"].unique().tolist()

    #Array of index lengths
    arr_ind = []

    temp = pd.DataFrame(index = dates, columns = custIds)
    for name in custIds:
        endix = len(df[df["LCLid"]==name]["KWH"])
        if endix >= 400:
          arr_ind.append(endix)
          temp[name][:endix] = df[df["LCLid"]==name]["KWH"]

    finalTable = temp[0:min(arr_ind)].dropna(axis=1)

    '''
    #Make a new dataframe
    temp = pd.DataFrame(index = dates, columns = custIds[0:40])

    #Import energy consumption from df into the temp datafram
    for name in custIds[0:40]:
        curDates = df[df["LCLid"]==name]["Date"].tolist()
        temp[name][curDates] = df[df["LCLid"]==name]["KWH"]

    #Drop all rows that have NaN values
    finalTable = temp.dropna()
    '''

    finalTable.to_csv(csv_path + "Tidy_LCL_Data.csv")
    


def oversample(X, y):
    
    #Perform oversampling
    X_oversampled, y_oversampled = ADASYN().fit_resample(X, y)

    #Perform feature scaling

    sc = MinMaxScaler()
    #sc = StandardScaler()
    X_oversampled = sc.fit_transform(X_oversampled).tolist()

    return(X_oversampled, y_oversampled)