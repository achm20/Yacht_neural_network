import tensorflow as tf
import Inputs_specifier as inputs

# Initialise lists and import ynn
rmse_list = []
sample_list = []
from Yacht_neural_network import ynn

# Loop predicting for x number of runs
runs = inputs.runs
runs = runs + 1


def sample_loop():
    for sample in range (1, runs):
        tf.keras.backend.clear_session()
        rmse = ynn()
        sample_list.append(sample)
        rmse_list.append(rmse)
    return sample_list, rmse_list

#print(sample_list)
#print(rmse_list)

