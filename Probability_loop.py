# Import libraries #
import tensorflow as tf
import Inputs_specifier as inputs
import Yacht_neural_network as ynn

# Initialise lists #
rmse_list = []
n_epochs_list = []

# Runs value clean up #
runs = inputs.runs
runs = runs + 1

# Function that loops for specified number of sample runs for a particular batch size.
# sample_loop runs function ynn in the Yacht_neural_network script for each sample run and
# returns the rmse and n_epochs of all sample runs for a particular batch size in a list.#


def sample_loop(batch_size):
    for sample in range (1, runs):
        tf.keras.backend.clear_session()
        print('batch size =', batch_size, ', sample =', sample)
        result = ynn.ynn(batch_size = batch_size)
        rmse = result[0]
        n_epochs = result[1]
        rmse_list.append(rmse)
        n_epochs_list.append(n_epochs)
    return rmse_list, n_epochs_list

# Function that resets rmse_list and n_epochs_list at the end of sample_loop and data extraction #


def reset_list():
    if len(rmse_list) == inputs.runs:
        rmse_list.clear()
        n_epochs_list.clear()

# Test case #


if __name__ == '__main__':
    batch_size = 8
    sample_loop(batch_size = batch_size)
    print(rmse_list)
    print(n_epochs_list)


