from models import GPUHA

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

model = GPUHA()
nsubs, g, t, e, _ = model.train(train_data, verbose=False, gpu=False)
print(
    "Aligned train shape: ",
    np.shape(nsubs),
    " err: ",
    e,
    " time: ",
    t,
    " Shared space shape: ",
    np.shape(g),
)

nsubs, t, e, _ = model.test(test_data, verbose=False)
print("Aligned test  shape: ", np.shape(nsubs), " err: ", e, " time: ", t)

# we can play around with the neural network shape and activation functions

# net_shape defines the number of layers and the number of neurons in each layer of the neural network.
# activation is the specific activation function being used in each layer of the neural network.
# we can try different activation functions such as softmax, or tanh

model1 = GPUHA()
tic = time.time()
model1.train(train_data, verbose=False, gpu=False)
toc = time.time() - tic
model1.test(test_data)
X2 = model1.Xtrain
Y2 = model1.Xtest
G2 = model1.G

print("\nGPUHA, Shared Space Shape: ", np.shape(G2))
print("GPUHA, Feature Shape: ", np.shape(X2))
print("GPUHA, Error: ", np.mean(model1.Etrain), ", Runtime: ", toc)
error = 0
for yi in Y2:
    error += np.linalg.norm(G2 - yi) ** 2
print("GPUHA, Test Error:", error)
