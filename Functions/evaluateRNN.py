import os
import numpy as np
from tensorflow import keras

'''Function performs RNN
Parameters include:
    -Training_data:
    -Training labels: true label for each training vector represented 
                      by one-hot vector y(x) = 0 for honest and y(x) = 1 for malicious
    -N : Number of neurons/hidden units in each layer
    -K : Fraction of the training data to be used
    -I : Number of Epochs
'''
def evalRNN(X_train, y_train, X_test, y_test, N, K, I):
    
    
    samples, features = np.shape(X_train)

    gru_model = keras.Sequential()
    gru_model.add(keras.layers.GRU(N,dropout=0.2,return_sequences=True,reset_after = True, input_shape=(features,1)))
    gru_model.add(keras.layers.GRU(N,dropout=0.2,reset_after = True))
    gru_model.add(keras.layers.Dense(2, activation="softmax"))

    #loss and optimizer
    loss = keras.losses.SparseCategoricalCrossentropy()
    optim = keras.optimizers.Adamax(learning_rate =  0.001)
    metrics = ["accuracy"]

    #Compile the model
    gru_model.compile(optim, loss, metrics)

    #Fit the model
    gru_model.fit(np.array(X_train), 
                  np.array(y_train), 
                  epochs = I,
                  verbose = 'auto',
                  shuffle = True)
                  
    #Predict output
    y_pred = np.round(gru_model.predict(X_test))
    
    #for i in range(len(y_pred)):
      #print("Expected=%s, Predicted=%s" % (y_test[i],y_pred[i]))
      
    
    #Evaluate the model
    gru_model.evaluate(np.array(X_test),
                       np.array(y_test),
                       verbose = 'auto',
                       return_dict = False)
                       
    return(computeMetrics(y_test,y_pred))
 

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
    
    return(DR, FPR, HD)
     