# Specify inputs #
patience = 3
epochs = 1000
runs = 100


def master_method():
    import Probability_loop as pl
    import Probability_density_plotter as pdp
    for batch_size in range(0, 33, 2):
        pl.sample_loop()
        sample_list = pl.sample_list
        rmse_list = pl.rmse_list
        pdp.probability_plotter()


if __name__ == '__main__':
    master_method()



