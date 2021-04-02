README

Yacht_neural_network.py completes four main tasks:
1. Reads and preprocesses the data, including data splitting and feature scaling 
2. Creates the neural network architecture
3. Trains the neural network
4. Tests the neural network and outputs the number of epochs required for training and the rmse from testing

Probability_loop.py loops Yacht_neural_network a number of times. This number of times is assigned as the variable 'runs'. Hence multiple neural networks are created, trained and tested. This script returns a list of rmse called rmse_list and a list of number of epochs called n_epochs_list for the consecutive neural networks created, trained and tested. If the number of entries in the list is equal to 'runs', then the list is cleared.

Probability_density_plotter.py has two methods: probability_density_plotter and histogram_plotter. 
Method probability_density_plotter takes the rmse_list created by Probability_loop.py and plots probability density vs rmse
Method histogram_plotter takes the rmse_list created by Probability_loop.py and plots a histogram of number of neural network instances vs rmse

Data_

