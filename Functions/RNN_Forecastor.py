# Importing the libraries
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow import keras

def train_test_split(dataset, tstart, tend):
    train = dataset[tstart:tend]
    test = dataset[tend+1:]
    return np.array(train), np.array(test)

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def plot_predictions(test, predicted, my_path, filename):
    plt.clf()

    fig = plt.figure()
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("Predicition")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.legend()

    fig.savefig(my_path + filename)

def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {:.2f}.".format(rmse))


def RNN_forecastor(dataset, start_tind, end_tind, n_steps, my_path, filename):
    #Set seed for reproducible
    #tensorflow.random.set_seed(455)
    #np.random.seed(455)

    #Split up the dataset into training and testing
    training_set, test_set = train_test_split(dataset, start_tind, end_tind)


    #We must first standardize the training set to avoid any possible outliers
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set = training_set.reshape(-1, 1)
    training_set_scaled = sc.fit_transform(training_set)

    #We also need to split the training set into X_train (inputs) and y_train (outputs)
    #X_train will be a sequence based on a specific step size, 1-n_step, and the y_train will be the remainder of that 
    #sequence 1-n:end
    X_train, y_train = split_sequence(training_set_scaled,n_steps)

    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)

    #Create the GRU model and compile
    model_gru = keras.models.Sequential()
    model_gru.add(keras.layers.GRU(units=125, activation="tanh", input_shape=(n_steps, 1)))
    model_gru.add(keras.layers.Dense(units=1))

    model_gru.compile(optimizer="RMSprop", loss="mse")
    #model_gru.summary()

    #Batch size was originially 32 and epochs were originally 370
    model_gru.fit(X_train, y_train, epochs=10, batch_size=32)

    #Repeat processing and normalize the test set
    dataset_total = dataset[:]
    inputs = np.array(dataset_total[len(dataset_total) - len(test_set) - n_steps :])
    inputs = inputs.reshape(-1, 1)
    
    #Scaling
    inputs = sc.transform(inputs)

    # Split into samples
    X_test, y_test = split_sequence(inputs, n_steps)

    # Reshape
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Make prediction
    predicted_result = model_gru.predict(X_test)

    # Will need to invert the returned values to get them in the original input format
    predictions = sc.inverse_transform(predicted_result)

    #Plot the result
    plot_predictions(test_set,predictions, my_path, filename)

    #Print out the RMSE
    return_rmse(test_set, predictions)

    #Return the model's predictions
    return predicted_result


#smart_meter_data = pd.read_csv("London_SM_data_total.csv").iloc[0:,1].tolist()
# To only get the datapoints at 1 hour intervals
#smart_meter_data = [float(smart_meter_data[i]) for i in range(0,len(smart_meter_data),2)] 
#plt.plot(smart_meter_data)

#print("finished")

#RNN_forecastor(smart_meter_data, 0, round(0.8*len(smart_meter_data)), 24, "./Output/","EV_RNN_pred.png")
  
#dataset = pd.read_csv("InputData/DOE_EVData.csv").iloc[0,1:].tolist() 
#dataset = dataset*4
#plt.plot(dataset)
#plt.clf()

#start = 0
#end = round(0.8*len(dataset))
#n_step = 24

#RNN_forecastor(dataset, start, end, n_step, "./Output/","EV_RNN_pred.png")
