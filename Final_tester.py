# Import libraries #
import Yacht_neural_network as ynn
import tensorflow as tf

# Specify final batch size and clear all history #
final_batch_size = 2
tf.keras.backend.clear_session()

# Train and test the final ANN #
final_result = ynn.ynn_final(batch_size = final_batch_size)
print('final rmse =', final_result[0])
print('final n_epochs =', final_result[1])