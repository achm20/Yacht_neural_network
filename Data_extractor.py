# Import libraries #
import Inputs_specifier as inputs
import numpy as np
import pandas as pd
import statistics

# Constants from Inputs_specifier script #
runs = inputs.runs
end_batch_size = inputs.end_batch_size

# Initialise lists #
batch_size_list = []
rmse_mean_list = []
n_epochs_mean_list = []
rmse_median_list = []
n_epochs_median_list = []
rmse_stdev_list = []
n_epochs_stdev_list = []

# Creates table containing n_epochs and rmse of ANN samples for a certain batch size #


def table_creator(batch_size, n_epochs_list, rmse_list):
    # Create column of sample indexes #
    sample_list = [list(range(1, runs + 1, 1))]
    sample_list = np.reshape(sample_list, (runs, 1))
    # Convert n_epochs_list and rmse_list into columns #
    tc_n_epochs_list = np.reshape(n_epochs_list, (len(n_epochs_list), 1))
    tc_rmse_list = np.reshape(rmse_list, (len(rmse_list), 1))
    # Concatenate columns #
    tc_table = np.concatenate((sample_list, tc_n_epochs_list, tc_rmse_list), axis = 1)
    # Add titles and formatting #
    tc_table_titles = [['neural network sample', 'number of training epochs',
                       'root mean square error in testing']]
    tc_table = np.append(tc_table_titles, tc_table, axis = 0)
    tc_table = pd.DataFrame(tc_table)
    tc_table.to_csv('Tables/Batch size ' + str(batch_size), header = False, index = False)

# Creates table containing central tendency statistics for all batch sizes #


def central_tendency_calculator(batch_size, n_epochs_list, rmse_list):

    # Calculate central tendency values #
    rmse_mean = statistics.mean(rmse_list)
    n_epochs_mean = statistics.mean(n_epochs_list)
    rmse_median = statistics.median(rmse_list)
    n_epochs_median = statistics.median(n_epochs_list)
    rmse_stdev = statistics.stdev(rmse_list)
    n_epochs_stdev = statistics.stdev(n_epochs_list)

    # Append lists with every batch size value #
    batch_size_list.append(batch_size)
    rmse_mean_list.append(rmse_mean)
    n_epochs_mean_list.append(n_epochs_mean)
    rmse_median_list.append(rmse_median)
    n_epochs_median_list.append(n_epochs_median)
    rmse_stdev_list.append(rmse_stdev)
    n_epochs_stdev_list.append(n_epochs_stdev)

    # Make complete central tendency table after evaluating all batch sizes #
    if batch_size == end_batch_size:
        ctc_table_titles = [['batch size', 'mean of the root mean square errors',
                           'median of the root mean square errors',
                           'sd of the root mean square errors',
                           'mean number of training epochs', 'median number of training epochs',
                           'sd of the number of training epochs']]

        ctc_table = np.column_stack((np.array(batch_size_list), np.array(rmse_mean_list),
                                     np.array(rmse_median_list), np.array(rmse_stdev_list),
                                     np.array(n_epochs_mean_list), np.array(n_epochs_median_list),
                                     np.array(n_epochs_stdev_list)))
        print(ctc_table)
        ctc_table = np.append(ctc_table_titles, ctc_table, axis = 0)
        ctc_table = pd.DataFrame(ctc_table)
        ctc_table.to_csv('Tables/CTC table', header=False, index=False)

# Test case#


if __name__ == '__main__':
    batch_size = 4
    n_epochs_list = [23, 56, 32, 76, 55, 34, 23, 87, 56, 12]
    rmse_list = [45.2, 11.7, 88.4, 54.9, 55.9, 33.4, 11.5, 87.5, 83.4, 81.3]
    table_creator(batch_size = batch_size, n_epochs_list = n_epochs_list, rmse_list = rmse_list)