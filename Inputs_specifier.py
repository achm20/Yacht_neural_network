# Specify inputs #
patience = 5
epochs = 1000
runs = 50
end_batch_size = 100
batch_size_values = [1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, end_batch_size]

# Master method drives entire program #


def master_method():
    # Import scripts #
    import Probability_loop as pl
    import Probability_density_plotter as pdp
    import Data_extractor as de
    # Loop for different batch size values #
    for batch_size in batch_size_values:
        # Run sample_loop function in Probability_loop script #
        sample_loop_result = pl.sample_loop(batch_size = batch_size)
        rmse_list = sample_loop_result[0]
        # Run probability_density_plotter function in Probability_density_plotter script #
        pdp.probability_density_plotter(batch_size = batch_size, rmse_list = rmse_list)
        # Run histogram_plotter function in Probability_density_plotter script #
        pdp.histogram_plotter(batch_size = batch_size, rmse_list = rmse_list)
        n_epochs_list = sample_loop_result[1]
        # Run table_creator function in Data_extractor script #
        de.table_creator(batch_size = batch_size, rmse_list = rmse_list, n_epochs_list =
        n_epochs_list)
        # Run central_tendency_calculator function in Data_extractor script #
        de.central_tendency_calculator(batch_size = batch_size, rmse_list = rmse_list,
                                       n_epochs_list = n_epochs_list)
        # Run reset_list function in Probability_loop script #
        pl.reset_list()

# Run master_method if this script is run directly #


if __name__ == '__main__':
    master_method()



