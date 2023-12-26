import os
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import MinMaxScaler

def preProcess(X, y):
    
    #Perform oversampling
    X_oversampled, y_oversampled = ADASYN().fit_resample(X, y)

    #Perform feature scaling

    sc = MinMaxScaler(feature_range=(0, 1))
    X_oversampled = sc.fit_transform(X_oversampled).tolist()

    return(X_oversampled, y_oversampled)