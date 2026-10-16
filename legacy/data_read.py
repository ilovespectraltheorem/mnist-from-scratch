import struct
import numpy as np

def image_extract(file_path, image_num):
    with open (file_path, "rb") as mnist_file:
        mnist_file.seek(0)
        header = struct.unpack(">IIII",mnist_file.read(16))
        body = mnist_file.read()
        body_array = np.frombuffer(body,np.uint8).reshape(image_num,28,28)
        return body_array

def label_extract(file_path):
    with open (file_path, "rb") as mnist_file:
        mnist_file.seek(0)
        header = struct.unpack(">II",mnist_file.read(8))
        body = mnist_file.read()
        body_array = np.frombuffer(body,np.uint8)
        return body_array

def data_extract():   
    train_dataset = image_extract("train-images.idx3-ubyte",60000)
    train_label = label_extract("train-labels.idx1-ubyte")
    test_dataset = image_extract("t10k-images.idx3-ubyte", 10000)
    test_label = label_extract("t10k-labels.idx1-ubyte")
    X = train_dataset.reshape(train_dataset.shape[0],-1).astype(np.float32)/255.0
    Y = train_label.astype(np.uint8)
    X_test = test_dataset.reshape(test_dataset.shape[0],-1).astype(np.float32)/255.0
    Y_test = test_label.astype(np.uint8)
    return X,Y,X_test,Y_test