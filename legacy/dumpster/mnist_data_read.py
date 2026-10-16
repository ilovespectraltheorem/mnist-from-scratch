import struct
import numpy as np
import matplotlib.pyplot as plt

def data_read(file_path, image_num):
    with open (file_path, "rb") as mnist_file:
        mnist_file.seek(0)
        header = struct.unpack(">IIII",mnist_file.read(16))
        body = mnist_file.read()
        body_array = np.frombuffer(body,np.uint8).reshape(image_num,28,28)
        return body_array

def label_read(file_path):
    with open (file_path, "rb") as mnist_file:
        mnist_file.seek(0)
        header = struct.unpack(">II",mnist_file.read(8))
        body = mnist_file.read()
        body_array = np.frombuffer(body,np.uint8)
        return body_array
        
train_dataset = data_read("/Users/jiangshuishan/Downloads/MNIST_ORG/train-images.idx3-ubyte",60000)
train_label = label_read("/Users/jiangshuishan/Downloads/MNIST_ORG/train-labels.idx1-ubyte")

test_dataset = data_read("/Users/jiangshuishan/Downloads/MNIST_ORG/t10k-images.idx3-ubyte", 10000)
test_label = label_read("/Users/jiangshuishan/Downloads/MNIST_ORG/t10k-labels.idx1-ubyte")

X = train_dataset.reshape(train_dataset.shape[0],-1).astype(np.float32)/255.0
Y = train_label.astype(np.uint8)

X_test = test_dataset.reshape(test_dataset.shape[0],-1).astype(np.float32)/255.0
Y_test = test_label.astype(np.uint8)

def initialize():
    W1 = np.random.randn(784,128)*np.sqrt(1/784)
    W2 = np.random.randn(128,10)*np.sqrt(1/128)
    B1 = np.zeros(128)
    B2 = np.zeros(10)
    X_batch = X[0:100]
    Y_batch = Y[0:100]
    return W1, W2, B1, B2, X_batch, Y_batch

def initialize_x_y():
    j = np.random.randint(59900)
    X_batch = X[j:j+100]
    Y_batch = Y[j:j+100]
    return X_batch, Y_batch

def shuffle(X_batch, Y_batch):
    perm = np.random.permutation(100)
    X_batch = X_batch[perm]
    Y_batch = Y_batch[perm]
    return X_batch, Y_batch

def layer_1(W1, B1, X_batch):
    X_batch.reshape(100,784)
    Z1 = X_batch@W1+B1
    return Z1

def layer_2(W2, B2, A1):
    A1.reshape(100,128)
    Z2 = A1@W2+B2
    return Z2

def sigmoid(z):
    return 1/(1+np.exp(-z))

def softmax(Z):
    row_max = np.max(Z, axis = 1, keepdims=True)
    exp_scores = np.exp(Z-row_max)
    row_sums = np.sum(exp_scores, axis = 1, keepdims=True)
    return exp_scores/row_sums

def loss (A2, Y_batch):
    num_examples = Y_batch.shape[0]
    correct_probability = A2[np.arange(num_examples), Y_batch]
    correct_probability = np.clip(correct_probability, 1e-12, 1.0)
    loss = -np.mean(np.log(correct_probability))
    return loss

learning_rate = 0.1
epochs = 100000

W1, W2, B1, B2, X_batch, Y_batch = initialize()

for i in range(epochs):
    X_batch, Y_batch = initialize_x_y()
    X_batch, Y_batch = shuffle(X_batch, Y_batch)
    A1 = sigmoid(layer_1(W1, B1, X_batch))
    A2 = softmax(layer_2(W2,B2,A1))
    loss_value = loss(A2, Y_batch)
    print(loss_value)

    Y_onehot = np.eye(10)[Y_batch]
    dZ2 = A2 - Y_onehot
    dW2 = (A1.T @ dZ2) / 100
    dB2 = np.sum(dZ2, axis= 0)/100
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * A1 * (1-A1)
    dW1 = (X_batch.T @ dZ1) / 100
    dB1 = np.sum(dZ1, axis=0) / 100

    W1 -= learning_rate * dW1
    B1-= learning_rate * dB1
    W2 -= learning_rate * dW2
    B2 -= learning_rate * dB2

def test():
    Z1 = sigmoid(X_test @ W1 + B1)
    Z2 = Z1 @ W2 + B2
    A2 = softmax (Z2)
    predictions = np.argmax(A2, axis = 1)
    correct = (predictions == Y_test)
    accuracy = np.mean(correct)
    return accuracy

print(test())

import tkinter as tk
from PIL import Image, ImageDraw

def draw_and_predict():
    canvas_size = 280

    window = tk.Tk()
    window.title("Draw a digit")

    canvas = tk.Canvas(window, width=canvas_size, height=canvas_size, bg="black")
    canvas.pack()

    image = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(image)

    def paint(event):
        x, y = event.x, event.y
        radius = 8
        canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="white", outline="white")
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=255)

    canvas.bind("<B1-Motion>", paint)

    def predict():
        small_image = image.resize((28, 28))
        pixels = np.array(small_image).astype(np.float32) / 255.0
        X_input = pixels.reshape(1, 784)

        Z1 = X_input @ W1 + B1
        A1 = sigmoid(Z1)
        Z2 = A1 @ W2 + B2
        A2 = softmax(Z2)
        prediction = np.argmax(A2, axis=1)[0]

        result_label.config(text=f"Predicted digit: {prediction}")

    def clear():
        canvas.delete("all")
        draw.rectangle([0, 0, canvas_size, canvas_size], fill=0)
        result_label.config(text="Draw a digit and click Predict")

    tk.Button(window, text="Predict", command=predict).pack()
    tk.Button(window, text="Clear", command=clear).pack()
    result_label = tk.Label(window, text="Draw a digit and click Predict")
    result_label.pack()

    window.mainloop()

draw_and_predict()
