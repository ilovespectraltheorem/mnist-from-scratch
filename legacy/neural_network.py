import numpy as np

global input_size, hidden_size
batch_size = 100
learning_rate = 0.1

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size=10):
        self.W1 = np.random.randn(input_size,hidden_size)*np.sqrt(1/input_size)
        self.W2 = np.random.randn(hidden_size,output_size)*np.sqrt(1/hidden_size)
        self.B1 = np.zeros(hidden_size)
        self.B2 = np.zeros(output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def softmax(self, z):
        row_max = np.max(z, axis = 1, keepdims=True)
        exp_scores = np.exp(z-row_max)
        row_sums = np.sum(exp_scores, axis = 1, keepdims=True)
        return exp_scores/row_sums

    def layer_1(self, X_batch):
        self.X_batch = X_batch.reshape(batch_size, self.input_size)
        self.Z1 = self.X_batch @ self.W1 + self.B1
        A1 = self.sigmoid(self.Z1)
        return A1

    def layer_2(self, A1):
        self.A1 = A1.reshape(batch_size, self.hidden_size)
        self.Z2 = self.A1 @ self.W2 + self.B2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backwards(self, Y_batch):
        Y_onehot = np.eye(10)[Y_batch]
        self.dZ2 = self.A2 - Y_onehot
        self.dW2 = (self.A1.T @ self.dZ2) / batch_size
        self.dB2 = np.sum(self.dZ2, axis= 0)/batch_size
        self.dA1 = self.dZ2 @ self.W2.T
        self.dZ1 = self.dA1 * self.A1 * (1-self.A1)
        self.dW1 = (self.X_batch.T @ self.dZ1) / batch_size
        self.dB1 = np.sum(self.dZ1, axis=0) / batch_size

        self.W1 -= learning_rate * self.dW1
        self.B1 -= learning_rate * self.dB1
        self.W2 -= learning_rate * self.dW2
        self.B2 -= learning_rate * self.dB2

    def loss(self, Y_batch):
        num_examples = Y_batch.shape[0]
        correct_probability = self.A2[np.arange(num_examples), Y_batch]
        correct_probability = np.clip(correct_probability, 1e-12, 1.0)
        loss = -np.mean(np.log(correct_probability))
        return loss
    
    
