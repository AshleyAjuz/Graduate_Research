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
def evaluateRNN(X_train, y_train, X_test, y_test, N, K, I):
    
    samples, features = np.shape(X_train)

    gru_model = keras.models.Sequential()
    # Input layer
    gru_model.add(keras.layers.Input(shape=(features,1)))
    # GRU layer 1
    gru_model.add(keras.layers.GRU(
        N,
        activation = "tanh",
        recurrent_activation = "sigmoid",
        use_bias = True,
        kernel_initializer = 'glorot_uniform',
        dropout = 0.2,
        return_sequences = True,
    ))
    # GRU layer 2
    gru_model.add(keras.layers.GRU(
        N, 
        activation = "tanh",
        dropout = 0.2
        ))
  
   # Output layer
    gru_model.add(keras.layers.Dense(1, activation="softmax"))

    #loss and optimizer
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    optim = keras.optimizers.Adam(learning_rate =  0.001)
    metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.FalseNegatives()]

    #Compile the model
    gru_model.compile(optim, loss, metrics)

    #Fit the model
    gru_model.fit(np.array(X_train), 
                  np.array(y_train), 
                  epochs = I,
                  verbose = 'auto',
                  validation_split = K,
                  shuffle = True)
    
    #Evaluate the model
    gru_model.evaluate(X_test,
                       y_test,
                       return_dict = False)
    
    #Compute Metrics

    #Assign the BinaryAcc metric to DR and the FalseNegative metric to FA
    metric_res = gru_model.get_metrics_result()
    DR = metric_res["accuracy"]
    FA = metric_res[""]
    HD = DR - FA

    return (DR, FA, HD)


    