from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.python.keras.utils import np_utils
import tensorflow as tf
from numpy.random import seed
from mlxtend.data import loadlocal_mnist
import os.path


class FashionClassifier(object):
    def __init__(self, train=False, data_path='fashion', SEED=123, model_path='cnn_fashion_model2.ckpt'):
        self.model = Sequential([
            Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)),
            MaxPool2D(pool_size=(1, 1)),
            Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)),
            MaxPool2D(pool_size=(1, 1)),
            Flatten(),
            Dense(100, activation='relu'),
            Dense(10, activation='softmax')
        ])

        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        if train:
            try:
                self.model.train(data_path, SEED)
            except Exception:
                print("I couldn't load your data, so I used my classifier.\n")
                self.model_load('cnn_fashion_model2.ckpt')
        else:
            if os.path.exists(model_path + '.index'):
                self.model_load(model_path)
            else:
                print("I couldn't load your classifier, so I used mine.\n")
                self.model_load('cnn_fashion_model2.ckpt')

    @staticmethod
    def prep_data(data_path):
        X_train, y_train = loadlocal_mnist(
            images_path=data_path + '/train-images-idx3-ubyte',
            labels_path=data_path + '/train-labels-idx1-ubyte')
        num_train_images = X_train.shape[0]
        X_train = X_train.reshape((num_train_images, 28, 28, 1)).astype('float32')

        X_test, y_test = loadlocal_mnist(
            images_path=data_path + '/t10k-images-idx3-ubyte',
            labels_path=data_path + '/t10k-labels-idx1-ubyte')
        num_test_images = X_test.shape[0]
        X_test = X_test.reshape((num_test_images, 28, 28, 1)).astype('float32')

        X_train /= 255
        X_test /= 255

        n_classes = 10
        Y_train = np_utils.to_categorical(y_train, n_classes)
        Y_test = np_utils.to_categorical(y_test, n_classes)

        return X_train, X_test, Y_train, Y_test

    def model_train(self, data_path, SEED):
        seed(SEED)
        tf.random.set_seed(SEED)

        X_train, X_test, Y_train, Y_test = self.prep_data(data_path)

        self.model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))

    def model_load(self, model_path):
        self.model.load_weights(model_path)
