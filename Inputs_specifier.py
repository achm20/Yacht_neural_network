# Specify inputs #
patience = 3
epochs = 1000
runs = 100


def master_method():
    import Probability_loop as pl
    for batch_size in range(2, 33, 2):
        pl.sample_loop(batch_size)
        sample_list = pl.sample_list
        rmse_list = pl.rmse_list


if __name__ == '__main__':
    master_method()



