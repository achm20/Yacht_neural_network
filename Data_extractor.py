import Inputs_specifier as inputs
import numpy as np
import pandas as pd

runs = inputs.runs


def table_creator(batch_size, n_epochs_list, rmse_list):
    data = [list(range(1, runs + 1, 1))]
    data.append(n_epochs_list)
    data.append(rmse_list)
    data = np.array(data).reshape(3, runs)
    table = pd.DataFrame(data)
    table.to_csv('Tables/Batch size ' + str(batch_size), header = False, index = False)


if __name__ == '__main__':
    batch_size = 8
    n_epochs_list = [23, 56, 32, 76, 55, 34, 23, 87, 56, 12]
    rmse_list = [45.2, 11.7, 88.4, 54.9, 55.9, 33.4, 11.5, 87.5, 83.4, 81.3]
    table_creator(batch_size = batch_size, n_epochs_list = n_epochs_list, rmse_list = rmse_list)