from neural_network import NeuralNetwork
from data_read import data_extract
import numpy as np
epochs = 10000
batch_size = 100
learning_rate = 0.1

network_0 = NeuralNetwork(784,128)
print(network_0.input_size)
epochs = 10000
batch_size = 100
learning_rate = 0.1
X,Y,X_test,Y_test = data_extract()

def initialize_x_y():
        j = np.random.randint(59900)
        X_batch = X[j:j+100]
        Y_batch = Y[j:j+100]
        perm = np.random.permutation(100)
        X_batch = X_batch[perm]
        Y_batch = Y_batch[perm]
        return X_batch, Y_batch

def train():
    for _ in range(epochs):
        X_batch, Y_batch = initialize_x_y()
        network_0.A1 = network_0.layer_1(X_batch)
        network_0.A2 = network_0.layer_2(network_0.A1)
        loss_value = network_0.loss(Y_batch)
        network_0.backwards(Y_batch)
        print(loss_value)

train()

def test():
    A1 = network_0.sigmoid(X_test @ network_0.W1 + network_0.B1)
    Z2 = A1 @ network_0.W2 + network_0.B2
    A2 = network_0.softmax (Z2)
    predictions = np.argmax(A2, axis = 1)
    correct = (predictions == Y_test)
    accuracy = np.mean(correct)
    return accuracy

print(test())
