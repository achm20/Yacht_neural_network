README

Yacht_neural_network.py completes four main tasks:
1. Reads and preprocesses the data, including data splitting and feature scaling 
2. Creates the neural network architecture
3. Trains the neural network
4. Tests the neural network (either using validation set if run by Inputs_specifier.py or test set Final_tester.py) and outputs the number of epochs required for training and the rmse from testing

Probability_loop.py loops Yacht_neural_network a number of times. This number of times is assigned as the variable 'runs'. Hence multiple neural networks are created, trained and tested. This script returns a list of rmse called rmse_list and a list of number of epochs called n_epochs_list for the consecutive neural networks created, trained and tested. If the number of entries in the list is equal to 'runs', then the list is cleared.

Probability_density_plotter.py has two methods: probability_density_plotter and histogram_plotter. 
Method probability_density_plotter takes the rmse_list created by Probability_loop.py and plots probability density vs rmse
Method histogram_plotter takes the rmse_list created by Probability_loop.py and plots a histogram of number of neural network instances vs rmse

Data_extractor.py has two methods: table_creator and central_tendency_calculator.
table_creator creates a table containing the rmse and number of epochs for each run
central_tendency_calculator creates a table containing the mean, median and standard deviation for rmse and number of epochs across a range of batch sizes.

Inputs_specifier.py is the main driving script and should be run for validation. In the following order, Inputs_specifier: 
1. Specifies list of batch sizes for validation
2. Loops the following for each value of batch size in the list of batch sizs:
   a. Probability_loop.py to obtain rmse_list and n_epochs_list for each batch size
   b. Probability_density_plotter.py to create the probability density plot and histogram plot for each batch size
   c. table_creator in Data_extractor.py to create the table of results for each batch size
3. Creates the central tendency table by running central_tendency_calculator in Data_extractor.py at the end

The results should give the most desirable batch size. This value should be assigned as final_batch_size in Final_tester.py.
Final_tester.py runs Yacht_neural_network to undertake final testing on the chosen batch size. 

