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
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
import json
import sys
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

# In case CUDA environment vars get messed up, Terminal commands to fix:
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
# export CUDA_HOME=/usr/local/cuda

# some other useful flags
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 32, "The batch size.")
flags.DEFINE_integer('lr', 0.0001, "The learning rate.")

DATA_FOLDER = 'driving_data/7/'
CSV_NAME = 'driving_log.csv'

# For folder 3
PATH_TO_FIX ='/home/casey/Desktop/'

with open(DATA_FOLDER + CSV_NAME, 'r') as f:
  reader = csv.reader(f)
  your_list = list(reader)
  # Limit for memory purposes
  # your_list = your_list[:2000]

# Get columns of .csv
center_imgs = [item[0] for item in your_list]
left_imgs = [item[1] for item in your_list]
right_imgs = [item[2] for item in your_list]
angle = [item[3] for item in your_list]
throttle = [item[4] for item in your_list]
brake = [item[5] for item in your_list]
speed = [item[6] for item in your_list]


def main(_):
    X = imglist_to_np(center_imgs)
    # X = grayscale(X)
    # X = normalize_img(X)
    # plt.imshow(X[0])
    # plt.show()
    y = np.asarray(angle)
    X, y = shuffle(X, y, random_state=0)
    X_train, X_val, X_test, y_train, y_val, y_test = make_sets(X, y)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen.fit(X_train)



    # Some info
    image_dim = X_train[0].shape
    input_dim = image_dim[0] * image_dim[1] * image_dim[2]

    # Reshape inputs, if starting with a Dense layer
    len_X_train = len(X_train)
    len_X_test = len(X_test)
    len_X_val = len(X_val)

    # comment out the reshape if using CNN
    #X_train = X_train.reshape(len_X_train, input_dim)
    #X_test = X_test.reshape(len_X_test, input_dim)
    #X_val = X_val.reshape(len_X_val, input_dim)

    dropout_1 = 0.5

    # some dimension info
    input_shape = image_dim
    print("imagedim: {}".format(image_dim))

    # Layer params

    # conv 1
    kernel_size1 = (5, 5)
    nb_filters1 = 24

    # conv 2
    kernel_size2 = (5, 5)
    nb_filters2 = 36

    # conv 3
    kernel_size3 = (5, 5)
    nb_filters3 = 48

    # conv 4
    kernel_size4 = (3, 3)
    nb_filters4 = 60

    # conv 5
    kernel_size5 = (3, 3)
    nb_filters5 = 60

    # Dense layers
    # NVIDIA paper: final layer is Dense, 1 neuron
    neurons1 = 1164
    neurons2 = 100
    neurons3 = 50
    neurons4 = 10
    neurons5 = 1

    model = Sequential()
    model.add(Convolution2D(nb_filters1, kernel_size1[0], kernel_size1[1],
                            border_mode='same',
                            activation='tanh',
                            input_shape=input_shape))
    model.add(Convolution2D(nb_filters2, kernel_size2[0], kernel_size2[1],
                            border_mode='same',
                            activation='tanh'))
    model.add(Convolution2D(nb_filters3, kernel_size3[0], kernel_size3[1],
                            border_mode='same',
                            activation='tanh'))
    model.add(Convolution2D(nb_filters4, kernel_size4[0], kernel_size4[1],
                            border_mode='same',
                            activation='tanh'))
    model.add(Convolution2D(nb_filters5, kernel_size5[0], kernel_size5[1],
                            border_mode='same',
                            activation='tanh'))
    model.add(Flatten())
    model.add(Dense(neurons1, activation='tanh'))
    model.add(Dropout(dropout_1))
    model.add(Dense(neurons2, activation='tanh'))
    model.add(Dense(neurons3, activation='tanh'))
    model.add(Dense(neurons4, activation='tanh'))
    model.add(Dense(neurons5))

    adam = Adam(lr=FLAGS.lr)
    rmsprop = RMSprop(lr=FLAGS.lr)
    # start with old weights from a good model
    # model.load_weights('model_start.h5')
    model.compile(loss='mse', optimizer=adam) # adam or rmsprop

    start_time = time.time()

    model.fit_generator(datagen.flow(X_train, y_train, batch_size=FLAGS.batch_size),
                        samples_per_epoch=len(X_train), nb_epoch=FLAGS.epochs,
                        validation_data=(X_val, y_val))
    #model.fit(X_train, y_train, verbose=1, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size,
    #          validation_data=(X_val, y_val), shuffle=True)
    print("Total training time on {} images: {} sec".format(len(X_train), time.time() - start_time))
    # Uncomment at end to get test scores
    #score = model.evaluate(X_test, y_test, batch_size=16, verbose=1)
    #print("\nTest score: {}".format(score))

    # save model to JSON
    model_json = model.to_json()
    with open('model.json', 'w') as f:
        json.dump(model_json, f)
    print("Saved model to JSON")
    # save weights to HDF5
    model.save_weights("model.h5")
    print("Saved weights to HDF5")


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

    X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.10)
    X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.10)

    print("Length of X_train: {}\n".format(len(X_train)))
    print("Length of X_val: {}\n".format(len(X_val)))
    print("Length of X_test: {}".format(len(X_test)))
    return X_train, X_val, X_test, y_train, y_val, y_test


# Fixes image paths after moving folders, resizes image to 32 x 16
def imglist_to_np(img_list):
    features = []
    count = 0
    for imgpath in img_list:
        count += 1
        sys.stdout.write("\rLoading image {} of {}".format(count, len(img_list)))
        sys.stdout.flush()
        newpath = imgpath.replace(PATH_TO_FIX,
                                  DATA_FOLDER)
        im = cv2.imread(newpath)
        im_resized = imresize(im, size=0.1)
        features.append(im_resized.astype(dtype='float64', casting='unsafe'))
    return np.array(features)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()