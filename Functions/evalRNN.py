import numpy as np
import tensorflow as tf
from tensorflow import keras


def evalRNN(X_train, y_train, X_test, y_test, N, B, I):
    
    X = np.array(X_train)
    y = np.array(y_train)
    X_t = np.array(X_test)
    y_t = np.array(y_test)
    
    #X, X_t = X[:,0:1], X_t[:,0:1] 
    
    #print(np.shape(X))
    #print(np.shape(X_t))

    gru_model = keras.Sequential()
    gru_model.add(keras.layers.GRU(N,dropout=0.2,return_sequences=True,reset_after = True, input_shape=(X.shape[1],1)))
    gru_model.add(keras.layers.GRU(N,dropout=0.2,return_sequences=True,reset_after = True))
    gru_model.add(keras.layers.GRU(N,dropout=0.2,return_sequences=True,reset_after = True))
    gru_model.add(keras.layers.GRU(N,dropout=0.2,reset_after = True))
    gru_model.add(keras.layers.Dense(2, activation="softmax"))

    #loss and optimizer
    loss = keras.losses.SparseCategoricalCrossentropy()
    optim = keras.optimizers.Adamax(learning_rate =  0.001)
    metrics = ["accuracy"]

    #Compile the model
    gru_model.compile(optim, loss, metrics)
      

    #Fit the model
    gru_model.fit(X, 
                  y, 
                  shuffle = False,
                  batch_size = B,
                  epochs = I,
                  verbose = 1,
                  )
                  
    #Predict output
    y_pred = np.round(gru_model.predict(X_t, batch_size = B, verbose = 1))
    
    '''
    for i in range(len(y_pred)):
      pred = list(y_pred[i]).index(1.0)
      print("Expected=%s, Predicted=%s" % (y_test[i],pred)) 
    '''

                       
    return(computeMetrics(y_t,y_pred))
 

def computeMetrics(y_test, y_pred):

    #First, transform y_pred format into same format as X_test
    
    ynew = []
    
    for i in range(len(y_pred)):
      ynew.append(list(y_pred[i]).index(1.0))
      
      
    #Compute True Positive (TP), False Positive (FP), False Negative (FN), and True Negative (TN) rates
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    for j in range(len(ynew)):
      if(ynew[j] ==1 and y_test[j] == 1):
        TP = TP + 1
      elif(ynew[j] ==1 and y_test[j] == 0):
        FP = FP + 1
      elif(ynew[j] ==0 and y_test[j] == 1):
        FN = FN + 1 
      elif(ynew[j] ==0 and y_test[j] == 0):
        TN = TN + 1    

    
    #Compute DR, FPR, and HD
    DR = float(TP/(TP + FP))
    FPR = float(FP/(FP + TN))
    HD = abs(DR - FPR)
    Acc = float((TP+TN)/(TP + FP + FN + TN))
    
    return(DR, FPR, HD, Acc)
    