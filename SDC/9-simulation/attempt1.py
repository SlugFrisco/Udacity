import pickle
import tensorflow as tf
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from scipy.misc import imresize
# import Keras layers you need here
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential

flags = tf.app.flags
FLAGS = flags.FLAGS

# some other useful flags
flags.DEFINE_integer('epochs', 10, "The number of epochs.")
flags.DEFINE_integer('batch_size', 1, "The batch size.")

DATA_FOLDER = 'driving_data/2/'
CSV_NAME = 'driving_log.csv'

with open(DATA_FOLDER + CSV_NAME, 'r') as f:
  reader = csv.reader(f)
  your_list = list(reader)
  your_list = your_list[1:1000]

# Get columns of .csv
center_imgs = [item[0] for item in your_list]
left_imgs = [item[1] for item in your_list]
right_imgs = [item[2] for item in your_list]
angle = [item[3] for item in your_list]
throttle = [item[4] for item in your_list]
brake = [item[5] for item in your_list]
speed = [item[6] for item in your_list]

test_batch_size = 100
learning_rate = 0.01

# let's start with something basic just to get a working network
# then try the NVIDIA architecture
# then try some transfer learning


def main(_):
    X = imglist_to_np(center_imgs)
    #dataset_size = len(X)
    #TwoDim_X = X.reshape(dataset_size, -1)
    # X = grayscale(X)
    X = normalize_img(X)
    # X = normalize(TwoDim_X)
    # plt.imshow(X[0])
    # plt.show()
    y = angle
    X_train, X_val, X_test, y_train, y_val, y_test = make_sets(X, y)
    # Turn lists of images into arrays
    # X_train = np.array(X_train)
    # X_val = np.array(X_val)
    # X_test = np.array(X_test)

    # Some info
    image_dim = X_train[0].shape
    input_dim = image_dim[0] * image_dim[1] * image_dim[2]

    # Reshape inputs
    len_X_train = len(X_train)
    len_X_test = len(X_test)
    len_X_val = len(X_val)
    print(len_X_test)

    # comment out the reshape for the CNN
    #X_train = X_train.reshape(len_X_train, input_dim)
    #X_test = X_test.reshape(len_X_test, input_dim)
    #X_val = X_val.reshape(len_X_val, input_dim)

    # set up the keras NN
    # no need to define number of classes: one output in final layer
    nb_classes = 1
    n_neurons_1 = 64
    dropout_1 = 0.5

    # some info
    img_rows = image_dim[0]
    img_cols = image_dim[1]
    input_shape = image_dim
    print("imagedim: {}".format(image_dim))

    # some params

    # conv kernel size
    kernel_size = (3, 3)
    nb_filters = 32

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Dense(n_neurons_1, input_shape=(input_dim,), name="hidden1"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_1))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax', name="output"))

    model.summary()  # hey what does this do?
    # Final output layer is one neuron. So use MSE as loss function instead of cross-entropy
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # train model
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val),
              shuffle=True)

    # train the NN, see validation accuracy each time

    # save network and weights
    # hook it up to the sim
    # try it with 3 images instead of just one


def grayscale(X):
    image_list = []
    for image in X:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_list.append(image)
    return np.array(image_list)


def normalize_img(image_list):
        normalized_train = []
        max_val = np.max(image_list)
        min_val = np.min(image_list)
        halfway = (max_val - min_val) / 2
        for i in range(len(image_list)):
            normalized_train.append(((image_list[i]) - halfway) / (max_val - min_val))
        return np.asarray(normalized_train)


def make_sets(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33)

    print("Length of X_train: {}".format(len(X_train)))
    print("Length of X_val: {}".format(len(X_val)))
    print("Length of X_test: {}".format(len(X_test)))
    return X_train, X_val, X_test, y_train, y_val, y_test

# Also resizes image to 50% of original
def imglist_to_np(img_list):
    features = []
    for imgpath in img_list:
        newpath = imgpath.replace('/home/casey/Desktop/Udacity-sandbox/',
                                  DATA_FOLDER)
        print(newpath)
        im = cv2.imread(newpath)
        im_resized = imresize(im, size=0.1)
        features.append(im)
    return np.array(features)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()