# Neural Network to predict residuary resistance per unit weight of displacement from features #

# Import libraries #
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Data preprocessing #
dataset = pd.read_csv('yacht_hydrodynamics.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into training set and test set #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.125, random_state
= 1)
# training size = 0.7, test size = 0.2, validation size = 0.1

# Feature scaling #
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Import Inputs_specifier values #
import Inputs_specifier as inputs

# This callback will stop training when there is no improvement in loss for 10 consecutive epochs
callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = inputs.patience)


# Building the ANN #
def ynn(batch_size):

    ann = tf.keras.models.Sequential()

    ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
    ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
    ann.add(tf.keras.layers.Dense(units = 1))

    # Training the ANN #
    ann.compile(optimizer = 'adam', loss = 'huber')

    history = ann.fit(X_train, y_train, batch_size = batch_size, epochs = inputs.epochs,
                      callbacks = [callback])

    n_epochs = len(history.history['loss'])
    print(n_epochs)

    y_pred = ann.predict(X_test)
    results = np.concatenate((np.reshape(y_pred, (len(y_pred), 1)), np.reshape(y_test, (len(y_test),
                                                                                       1))),axis = 1)
    #print(results)

    rmse = []
    for row in range(0, len(y_pred)):
        rmse.append((results[row, 0] - results[row, 1])**2)

    rmse = np.sqrt(sum(rmse))
    #print(rmse)
    return rmse, n_epochs


if __name__ == '__main__':
    ynn(batch_size = 8)