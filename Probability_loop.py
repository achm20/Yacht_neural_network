import tensorflow as tf
import Inputs_specifier as inputs
import Yacht_neural_network as ynn

# Initialise lists and import ynn
rmse_list = []
sample_list = []

# Loop predicting for x number of runs
runs = inputs.runs
runs = runs + 1


def sample_loop(batch_size):
    for sample in range (1, runs):
        tf.keras.backend.clear_session()
        rmse = ynn.ynn(batch_size = batch_size)
        sample_list.append(sample)
        rmse_list.append(rmse)
    return sample_list, rmse_list


sample_loop()
print(sample_list, rmse_list)


