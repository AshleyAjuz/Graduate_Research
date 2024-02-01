import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def preProcess(df):
    #Get all of the customer IDs
    custIds = df["LCLid"].unique().tolist()

    #Get all unique Dates
    dates = df["Date"].unique().tolist()

    #Make a new dataframe
    temp = pd.DataFrame(index = dates, columns = custIds)

    #Import energy consumption from df into the temp datafram
    for name in custIds:
        curDates = df[df["LCLid"]==name]["Date"].tolist()
        temp[name][curDates] = df[df["LCLid"]==name]["KWH"]

    print("Here")
    


def oversample(X, y):
    
    #Perform oversampling
    X_oversampled, y_oversampled = ADASYN().fit_resample(X, y)

    #Perform feature scaling

    sc = MinMaxScaler()
    #sc = StandardScaler()
    X_oversampled = sc.fit_transform(X_oversampled).tolist()

    return(X_oversampled, y_oversampled)