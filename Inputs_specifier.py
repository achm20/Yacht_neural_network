# Specify inputs #
patience = 3
epochs = 1000
runs = 10


def master_method():
    import Probability_loop as pl
    import Probability_density_plotter as pdp
    for batch_size in range(2, 5, 2):
        rmse_list = pl.sample_loop(batch_size = batch_size)
        pdp.probability_plotter(batch_size = batch_size, rmse_list = rmse_list)


if __name__ == '__main__':
    master_method()



