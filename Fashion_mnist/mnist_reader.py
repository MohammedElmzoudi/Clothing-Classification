import os
import gzip
import numpy as np

def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels_flat = np.expand_dims(np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8),axis=1)
        labels = np.zeros([10,len(labels_flat)])

    for i in range(len(labels_flat)):
        labels[:,i] = to_vector(labels_flat[i])
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels_flat), 784)

    return images, labels

def to_vector(i):
    output = np.zeros([10])
    output[i] = 1
    return output
