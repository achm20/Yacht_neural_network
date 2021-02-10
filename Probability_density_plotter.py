# Import libraries
import numpy as np
from scipy.stats.kde import gaussian_kde
from matplotlib import pyplot as plt


def probability_plotter(batch_size, rmse_list):
    kde = gaussian_kde(rmse_list)
    probability = np.linspace(min(rmse_list), max(rmse_list), 100)
    fig, sp = plt.subplots(1, 2, figsize = (23, 13))

    sp[0].plot(probability, kde(probability), 'tab:green')
    sp[0].set_title('Probability density plot of sum RMSE using Gaussian kernel')
    sp[0].set(xlabel = 'sum RMS error', ylabel = 'Probability')
    sp[1].hist(rmse_list, bins = 20)
    sp[1].set_title('Histogram plot of sum RMSE')
    sp[1].set(xlabel = 'sum RMS error', ylabel = 'Number of neural network instances')
    plt.suptitle('Batch size = ' + str(batch_size))
    plt.savefig('Plots/Batch size = ' + str(batch_size), format = 'png')


if __name__ == '__main__':
    probability_plotter()
    plt.show()