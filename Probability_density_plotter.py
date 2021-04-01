# Import libraries #
import numpy as np
from scipy.stats.kde import gaussian_kde
from matplotlib import pyplot as plt

# Probability density plotter function using rmse_list data #


def probability_density_plotter(batch_size, rmse_list):
    kde = gaussian_kde(rmse_list)
    probability = np.linspace(0, max(rmse_list) + 2, 100)
    plt.figure(figsize = (9, 7))
    plt.plot(probability, kde(probability), color = 'tab:green')
    plt.xlim([0, 4])
    plt.title('Probability Density Plot of Root Mean Square Error Using Gaussian Kernels - '
              'Batch '
              'Size ' +
              str(batch_size))
    plt.xlabel('Root mean square error')
    plt.ylabel('Probability density')
    plt.savefig('Plots/Batch size ' + str(batch_size) + ' probability density plot', format =
    'png')

# Histogram plotter function using rmse_list data #


def histogram_plotter(batch_size, rmse_list):
    plt.figure(figsize = (9, 7))
    plt.hist(rmse_list, bins = 20)
    plt.xlim([0, 4])
    plt.title('Histogram Plot of Root Mean Square Error - Batch Size ' + str(batch_size))
    plt.xlabel('Root mean square error')
    plt.ylabel('Number of neural network instances')
    plt.savefig('Plots/Batch size ' + str(batch_size) + ' histogram plot', format = 'png')

# Test case #


if __name__ == '__main__':
    batch_size = 2

    rmse_list = [68.6605682492256, 73.3942501938343, 82.73388053536414, 139.59889603137967,
                 115.54501253843304, 93.68302888035771, 89.51298006415365, 119.06000158309934,
                 72.55675334572791, 68.19579288244248, 70.78243602037429, 76.00915799379348,
                 50.113780162334436,
                 74.06191851973534, 75.73021152019498, 71.1260237276554, 77.70949514389036,
                 64.26126637458802, 71.97929924368859, 73.19325763106346, 69.36266697764398,
                 68.56743602871896, 138.61954211950302, 60.191937866210935, 81.30627345681194,
                 97.39331553459166, 72.70850450992586, 76.99120911717412, 121.90635196804996,
                 67.49022384643553, 81.48564965426925, 111.26983091831204, 69.41598737239835,
                 67.53477492809293, 96.19757166385645, 120.58487298846245, 41.841616013050086,
                 78.54278226971627, 87.22510704278946, 64.53768599390982, 77.96121495723725,
                 57.08768970012666, 65.44391728401185, 71.10606885194775, 108.02005864381786,
                 92.56713142156599, 68.17462605953216, 75.75972256183627, 95.8032851958275,
                 67.51007457137106]
    probability_density_plotter(batch_size = batch_size, rmse_list = rmse_list)
    histogram_plotter(batch_size = batch_size, rmse_list = rmse_list)
    plt.show()