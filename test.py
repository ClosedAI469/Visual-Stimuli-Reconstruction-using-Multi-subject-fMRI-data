from models import GPUHA, DHA

# from utils.easyfmri.Hyperalignment.DHA import DHA
import time
import numpy as np

import scipy.io as sio

mat = sio.loadmat("data/DS105-objectviewing-ROI-std4mm-DLY-TempAlign.mat")

data = mat["data"]
subject = mat["subject"]

data = data.reshape([6, 1452, 2294])

train_data = data[:-1]  # shape is (5, 1452, 2294)
test_data = data[-1:]  # shape is (1, 1452, 2294), one subject left out for testing
print(train_data.shape)
print(test_data.shape)

# %%
# we can play around with the neural network shape and activation functions

# net_shape defines the number of layers and the number of neurons in each layer of the neural network.
# activation is the specific activation function being used in each layer of the neural network.
# we can try different activation functions such as softmax, or tanh

ha = GPUHA()
tic = time.time()
ha.train(train_data, verbose=False, gpu=False)
toc = time.time() - tic
ha.test(test_data)
ha_transformed_train = ha.Xtrain
ha_transformed_test = ha.Xtest
ha_common_space = ha.G

print("\nGPUHA, Shared Space Shape: ", np.shape(ha_common_space))
print("GPUHA, Feature Shape: ", np.shape(ha_transformed_train))
print("GPUHA, Error: ", np.mean(ha.Etrain), ", Runtime: ", toc)
error = 0
for yi in ha_transformed_test:
    error += np.linalg.norm(ha_common_space - yi) ** 2
print("GPUHA, Test Error:", error)

# %%
# we can play around with the neural network shape and activation functions

dha = DHA([100, 50, 10], ["relu", "tanh", "softmax"], iteration=10, epoch=100)

# net_shape defines the number of layers and the number of neurons in each layer of the neural network.
# activation is the specific activation function being used in each layer of the neural network.
# we can try different activation functions such as softmax, or tanh

dha.train(train_data)
dha.test(test_data)
dha_transformed_train = dha.TrainFeatures
dha_transformed_test = dha.TestFeatures
dha_common_space = dha.Share

print(
    "\nDHA, trace(G) = ",
    np.trace(dha_common_space),
    " G^TG= ",
    np.trace(np.dot(np.transpose(dha_common_space), dha_common_space))
    / np.shape(dha_common_space)[0],
)
print("DHA, Shared Space Shape: ", np.shape(dha_common_space))
print("DHA, Features Shape: ", np.shape(dha_transformed_train))
print("DHA, Loss vec: ", dha.ha_loss_vec)
print("DHA, Error: ", dha.ha_loss, ", Runtime: ", dha.TrainRuntime)
# The error is L2 norm
error = 0
for yi in dha_transformed_test:
    error += np.linalg.norm(dha_common_space - yi) ** 2
print("DHA, Test Error:", error)
