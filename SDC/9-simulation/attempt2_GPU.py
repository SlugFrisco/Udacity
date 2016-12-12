import pickle
import tensorflow as tf
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from scipy.misc import imresize
# import Keras layers you need here
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD
import json

flags = tf.app.flags
FLAGS = flags.FLAGS

'''
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
'''

# some other useful flags
flags.DEFINE_integer('epochs', 20, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")

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
learning_rate = 1

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
    X, y = shuffle(X, y, random_state=0)
    X_train, X_val, X_test, y_train, y_val, y_test = make_sets(X, y)

    # Some info
    image_dim = X_train[0].shape
    input_dim = image_dim[0] * image_dim[1] * image_dim[2]

    # Reshape inputs
    len_X_train = len(X_train)
    len_X_test = len(X_test)
    len_X_val = len(X_val)
    # print(len_X_test)

    # comment out the reshape for the CNN
    #X_train = X_train.reshape(len_X_train, input_dim)
    #X_test = X_test.reshape(len_X_test, input_dim)
    #X_val = X_val.reshape(len_X_val, input_dim)

    # set up the keras NN
    # no need to define number of classes: one output in final layer
    nb_classes = 1
    n_neurons_1 = 1164
    dropout_1 = 0.5

    # some info
    img_rows = image_dim[0]
    img_cols = image_dim[1]
    input_shape = image_dim
    print("imagedim: {}".format(image_dim))

    # some params

    # conv 1
    kernel_size1 = (5, 5)
    nb_filters1 = 24

    # conv 2
    kernel_size2 = (5, 5)
    nb_filters2 = 36

    model = Sequential()
    model.add(Convolution2D(nb_filters1, kernel_size1[0], kernel_size1[1],
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam') # adam or rmsprop

    model.fit(X_train, y_train, nb_epoch=20, batch_size=16, verbose=1)
    #score = model.evaluate(X_test, y_test, batch_size=16, verbose=1)
    #print("\nTest score: {}".format(score))

    # serialize model to JSON
    model_json = model.to_json()
    with open('model.json', 'w') as f:
        json.dump(model_json, f)

    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


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

    X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.25)
    X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.33)

    print("Length of X_train: {}".format(len(X_train)))
    print("Length of X_val: {}".format(len(X_val)))
    print("Length of X_test: {}".format(len(X_test)))
    return X_train, X_val, X_test, y_train, y_val, y_test

# Also resizes image to 32 x 16 of original
def imglist_to_np(img_list):
    features = []
    for imgpath in img_list:
        newpath = imgpath.replace('/home/casey/Desktop/Udacity-sandbox/',
                                  DATA_FOLDER)
        #print(newpath)
        im = cv2.imread(newpath)
        im_resized = imresize(im, size=0.1)
        features.append(im_resized)
    return np.array(features)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()