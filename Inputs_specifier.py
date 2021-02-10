# Specify inputs #
patience = 2
epochs = 1000
runs = 5


def master_method():
    import Probability_loop as pl
    import Probability_density_plotter as pdp
    for batch_size in range(2, 5, 2):
        sample_loop_result = pl.sample_loop(batch_size = batch_size)
        rmse_list = sample_loop_result[0]
        pdp.probability_density_plotter(batch_size = batch_size, rmse_list = rmse_list)
        pdp.histogram_plotter(batch_size = batch_size, rmse_list = rmse_list)
        n_epochs_list = sample_loop_result[1]
        sample_list = [range(1, runs + 1, 1)]



if __name__ == '__main__':
    master_method()



