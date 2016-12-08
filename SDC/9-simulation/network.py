'''
Example usage:
python3

'''
import pickle
import tensorflow as tf
import numpy as np
# TODO: import Keras layers you need here
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

flags = tf.app.flags
FLAGS = flags.FLAGS

# some other useful flags
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val =

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    test_batch_size = 100
    learning_rate = 0.001

    nb_classes = len(np.unique(y_train)) # Get number of classes

    # Define input shape
    input_shape = X_train.shape[1:]
    # i.e. if the shape is (10000, 32, 32, 3) (so 10,000 in the training set, 32 x 32 image, 3 channel)
    # this allows us in take in (, 32, 32, 3), with the first unknown leaving us open to any batch size
    #                           (0, 1,  2, 3)
    input = Input(shape=input_shape)
    # REMEMBER TO FLATTEN!
    x = Flatten()(input)


    # Dense is a linear layer
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(input, x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # guessing the model has a prebaked dictionary key called accuracy

    # TODO: train your model here
    # train model
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val),
    shuffle=True)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
