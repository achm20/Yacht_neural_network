import Yacht_neural_network as ynn

batch_size = 20

final_result = ynn.ynn_final(batch_size = 20)
print('final rmse =', final_result[0])
print('final n_epochs =', final_result[1])