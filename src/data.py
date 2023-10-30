from torch.utils.data import Dataset
import numpy as np
from os.path import join
import struct
from array import array
import torch


class GenericDataset(Dataset):

    def __init__(self, split):
        self.split = split
        input_path = 'archive'
        training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

        loader = MnistDataloader(
            training_images_filepath,
            training_labels_filepath,
            test_images_filepath,
            test_labels_filepath
        )
        self.train, self.dev = loader.load_data()
        if self.split == 'train':
            self.data = self.train
            del self.dev
        if self.split == 'dev':
            self.data = self.dev
            del self.train

    def __getitem__(self, idx):
        zeros = torch.zeros(10)
        zeros[self.data[1][idx]] = 1
        y = zeros
        return np.asarray(self.data[0][idx]).ravel().astype('float32'), y

    def __len__(self):
        return len(self.data[0])

    def get_in_out_size(self):
        return np.asarray(self.data[0][0]).ravel().shape[-1], 10


class MnistDataloader(object):

    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img
        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)


def generic():
    ds = GenericDataset('dev')
    import ipdb; ipdb.set_trace()
    ds[0]


if __name__ == "__main__":
    generic()
