# Specify inputs #
patience = 2
epochs = 1000
runs = 10


def master_method():
    import Probability_loop as pl
    import Probability_density_plotter as pdp
    import Data_extractor as de
    for batch_size in range(2, 5, 2):
        sample_loop_result = pl.sample_loop(batch_size = batch_size)
        rmse_list = sample_loop_result[0]
        pdp.probability_density_plotter(batch_size = batch_size, rmse_list = rmse_list)
        pdp.histogram_plotter(batch_size = batch_size, rmse_list = rmse_list)
        n_epochs_list = sample_loop_result[1]
        de.table_creator(batch_size = batch_size, rmse_list = rmse_list, n_epochs_list = n_epochs_list)
        pl.reset_list()


if __name__ == '__main__':
    master_method()



