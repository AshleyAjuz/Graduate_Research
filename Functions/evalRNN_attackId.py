import numpy as np
import tensorflow as tf
from tensorflow import keras


def evalRNN_attackId(X_train, y_train, X_test, y_test, N, B, I):
    
    X = np.array(X_train)
    y = np.array(y_train)
    X_t = np.array(X_test)
    y_t = np.array(y_test)
    

    gru_model = keras.Sequential()
    gru_model.add(keras.layers.GRU(N,dropout=0.2,return_sequences=True,reset_after = True, input_shape=(X.shape[1],1)))
    gru_model.add(keras.layers.GRU(N,dropout=0.2,return_sequences=True,reset_after = True))
    gru_model.add(keras.layers.GRU(N,dropout=0.2,return_sequences=True,reset_after = True))
    gru_model.add(keras.layers.GRU(N,dropout=0.2,reset_after = True))
    gru_model.add(keras.layers.Dense(6, activation="softmax"))

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
    
    
    for i in range(len(y_pred)):
      pred = list(y_pred[i]).index(1.0)
      print("Expected=%s, Predicted=%s" % (y_test[i],pred)) 
    

                       
    return(computeMetrics(y_t,y_pred))
 

def computeMetrics(y_test, y_pred):

    #First, transform y_pred format into same format as y_test
    ynew = []
    
    for i in range(len(y_pred)):
      ynew.append(list(y_pred[i]).index(1.0))
    
    #Hashmap to store occurences of attacks
    attack_occ = {'0':0, '1':0, '2':0 , '3':0, '4':0, '5':0}
      
    #Define count (needed for computing accuracy)
    count = 0
    
    for j in range(len(ynew)):
      if(ynew[j] == y_test[j]):
         count += 1
         attack_occ[f"{y_test[j]}"] +=1

    
    #Compute Acc
    Acc = float(count/len(y_test))

    #Compute confusion matrix to store which attacks are detected the most
    cf = []
    for i in range(1,6):
        totalOcc = sum(np.array(y_test) == i)
        instancesFound = attack_occ[f"{i}"] 
        cf.append(float(instancesFound/totalOcc))

    print(f"The Accuracy of this detector is {Acc}")
    print(f"Attack 1:{cf[1]},Attack 2:{cf[2]}, Attack 3:{cf[3]}, Attack 4:{cf[4]}, Attack 5:{cf[5]} ")
        
    