import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_validate, y_validate = x_test[:-10], y_test[:-10]
x_test, y_test = x_test[-10:], y_test[-10:]

model = keras.Sequential()
model.add(keras.layers.GRU(64,dropout=0.2,return_sequences=True,reset_after = True, input_shape=(28, 28)))
model.add(keras.layers.GRU(64,dropout=0.2,reset_after = True))
model.add(keras.layers.Dense(10, activation="softmax"))


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adamax(learning_rate =  0.001),
    metrics=["accuracy"],
)	

model.fit(
    x_train, y_train, validation_data=(x_validate, y_validate), batch_size=64, epochs=10
)

for i in range(10):
    result = tf.argmax(model.predict(tf.expand_dims(x_test[i], 0)), axis=1)    
    print(result.numpy(), y_test[i])