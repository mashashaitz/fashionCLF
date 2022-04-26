from mlxtend.data import loadlocal_mnist
import os.path


class TestData(object):
    def __init__(self, path='fashion'):

        if os.path.exists(path + '/t10k-images-idx3-ubyte') and os.path.exists(path + '/t10k-labels-idx1-ubyte'):
            self.X_test, _ = loadlocal_mnist(
                images_path='fashion/t10k-images-idx3-ubyte',
                labels_path='fashion/t10k-labels-idx1-ubyte')
            num_test_images = self.X_test.shape[0]
            self.X_test = self.X_test.reshape((num_test_images, 28, 28, 1))

        else:
            print("We couldn't find your test data")
            self.X_test, _ = loadlocal_mnist(
                images_path='fashion/t10k-images-idx3-ubyte',
                labels_path='fashion/t10k-labels-idx1-ubyte')
            num_test_images = self.X_test.shape[0]
            self.X_test = self.X_test.reshape((num_test_images, 28, 28, 1))
