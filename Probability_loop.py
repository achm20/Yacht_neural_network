import tensorflow as tf
import Inputs_specifier as inputs
import Yacht_neural_network as ynn

# Initialise lists and import ynn
rmse_list = []
sample_list = []
n_epochs_list = []

# Loop predicting for x number of runs
runs = inputs.runs
runs = runs + 1


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


if __name__ == '__main__':
    batch_size = 8
    sample_loop(batch_size = batch_size)
    print(rmse_list)
    print(n_epochs_list)


