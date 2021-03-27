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

# Splitting the dataset into training, validation and test sets #
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 1)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.125,
                                                    random_state = 1)
# training size = 0.7, validation size = 0.2, test size = 0.1

# Feature scaling #
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Import Inputs_specifier values #
import Inputs_specifier as inputs

# This callback will stop training when there is no improvement in loss for patience consecutive
# epochs
callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = inputs.patience)


# Define function called ynn that creates an instance of the required ANN architecture, trains the
# ANN, prints number of epochs for training, tests the ANN on test data, prints the rmse of ANN
# during testing and returns rmse and n_epochs #
def ynn(batch_size):

    # Building the ANN architecture #
    ann = tf.keras.models.Sequential()

    # first hidden layer
    ann.add(tf.keras.layers.Dense(units = 6, input_dim = 6, activation = 'relu'))
    # second hidden layer
    ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
    # output layer
    ann.add(tf.keras.layers.Dense(units = 1))

    # Training the ANN #
    ann.compile(optimizer = 'adam', loss = 'huber')
    history = ann.fit(X_train, y_train, batch_size = batch_size, epochs = inputs.epochs,
                      callbacks = [callback])

    n_epochs = len(history.history['loss'])
    # print number of epochs used to train ANN
    print('number of epochs =', n_epochs)

    # Assess performance using validation set #
    y_pred = ann.predict(X_val)
    results = np.concatenate((np.reshape(y_pred, (len(y_pred), 1)), np.reshape(y_val, (len(y_val),
                                                                                       1))),axis = 1)
    # RMSE calculation #
    rmse = []
    for row in range(0, len(y_pred)):
        rmse.append((results[row, 0] - results[row, 1])**2)

    rmse = np.sqrt(sum(rmse)/len(y_pred))
    # print rmse for current ANN
    print('rmse =', rmse)
    return rmse, n_epochs

# If statement if this script is run directly #


if __name__ == '__main__':
    ynn(batch_size = 8)