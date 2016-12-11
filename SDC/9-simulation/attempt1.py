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
from keras.models import Model

flags = tf.app.flags
FLAGS = flags.FLAGS

# some other useful flags
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")

DATA_FOLDER = 'driving_data/2/'
CSV_NAME = 'driving_log.csv'

with open(DATA_FOLDER + CSV_NAME, 'r') as f:
  reader = csv.reader(f)
  your_list = list(reader)

# Get columns of .csv
center_imgs = [item[0] for item in your_list]
left_imgs = [item[1] for item in your_list]
right_imgs = [item[2] for item in your_list]
angle = [item[3] for item in your_list]
throttle = [item[4] for item in your_list]
brake = [item[5] for item in your_list]
speed = [item[6] for item in your_list]

test_batch_size = 100
learning_rate = 0.001

# let's start with something basic just to get a working network
# then try the NVIDIA architecture
# then try some transfer learning


def main(_):
    X = imglist_to_np(center_imgs)
    X = grayscale(X)
    X = normalize_img(X)
    # plt.imshow(X[0])
    # plt.show()
    y = angle
    X_train, X_val, X_test, y_train, y_val, y_test = make_sets(X, y)
    # Turn lists of images into arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)

    # set up the keras NN
    # no need to define number of classes: one output in final layer
    nb_classes = 1

    # Define input shape
    input_shape = X_train.shape[1:]
    input = Input(shape=input_shape)
    # REMEMBER TO FLATTEN!
    x = Flatten()(input)

    # Dense is a linear layer
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(input, x)
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
    return image_list


def normalize_img(X):
    image_list = []
    for image in X:
        image = normalize(image)
        image_list.append(image)
    return image_list


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
        newpath = imgpath.replace('/home/casey/Desktop/Udacity SDC Simulator/Default Linux desktop Universal_Data/',
                                  DATA_FOLDER)
        # print(newpath)
        im = cv2.imread(newpath)
        im_resized = imresize(im, size=0.5)
        features.append(im)
    return features

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()