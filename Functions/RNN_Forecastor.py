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


    fig = plt.figure()
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("Predicition")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.legend()


    fig.savefig(my_path + filename)




def evaluateMetrics(test, predicted):
    #Redefine the test and predicted sets
    test = test.tolist()
    predicted = predicted.tolist()


    #Define variables
    Acc = 0


    #Calculate the root mean squared erro
    #This will be the threshold value to compare the difference between the predicted and real results to
    rmse = np.sqrt(mean_squared_error(test[:len(predicted)], predicted))


    for j in range(len(predicted)):
        diff = abs(test[j][0] - predicted[j][0])
        if(diff <= rmse):
            Acc += 1
   
    Acc = float(Acc/len(predicted))
   
    print(f"The accuracy is {Acc} \n")
    print("The root mean squared error is {:.2f}.".format(rmse))


    return(rmse)


def isUnderAttack(test, predicted,rmse,thrs):
    isCompromised = False


    #Redefine the test and predicted sets
    test = test.tolist()
    predicted = predicted.tolist()


    for j in range(len(predicted)):
        diff = abs(test[j][0] - predicted[j][0])
        if(diff > rmse):
            count+=1
       
        if(count > thrs):
            isCompromised = True
            break
   
    return isCompromised






def RNN_forecastor(dataset, start_tind, end_tind, n_steps, my_path, filename):
    #Set seed for reproducible
    #tensorflow.random.set_seed(455)
    #np.random.seed(455)


    #Split up the dataset into training and testing
    training_set, test_set = train_test_split(dataset, start_tind, end_tind)


    training_set, test_set = training_set.T, test_set.T


    #Reshape the training dataset
    sc = MinMaxScaler(feature_range=(0,1))
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




    model_gru.fit(X_train, y_train, epochs=50, batch_size=32)


    #Repeat processing and normalize the test set
    '''
    dataset_total = dataset[:]
    inputs = np.array(dataset_total[len(dataset_total) - len(test_set) - n_steps :])
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    '''


    # Format test set so it is in the same format as the training data
    test_set = test_set.reshape(-1, 1)
    inputs = sc.fit_transform(test_set)
    X_test, y_test = split_sequence(inputs, n_steps)


    # Reshape
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


    # Make prediction
    predicted_results = model_gru.predict(X_test)


    #Plot the result
    plot_predictions(y_test,predicted_results, my_path, filename)


    #Print out the RMSE
    rmse = evaluateMetrics(y_test, predicted_results)


    #Return the model's predictions
    return (model_gru, rmse)