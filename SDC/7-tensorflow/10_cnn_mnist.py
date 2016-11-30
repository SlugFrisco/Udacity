# based on tutorial from https://github.com/Hvass-Labs/TensorFlow-Tutorials

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn
import time
from datetime import timedelta
import math

def main():
    optimize(epochs)


# Conv layer configurations
# -------------------------
# Conv layer 1
filter_size1 = 5    # 5x5
num_filters1 = 16   # use 16 of these filters

# Conv layer 2
filter_size2 = 5    # 5x5
num_filters2 = 36   # use 36 of these filters

# Fully connected
fc_size = 128       # use 128 neurons in this layer

# NN hyperparams:
learning_rate = 0.002
epochs = 1024
# Feeding all images through at once is too much
# Run one batch each time, pick this many images to send through
train_batch_size = 64
test_batch_size = 256
print_step = 8

# -------------------------

# Bring in the tensorflow MNIST data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot=True)

# Get some info about this dataset
print("Size of:")
print("Training set:\t\t{}".format(len(data.train.labels)))
print("Testing set:\t\t{}".format(len(data.test.labels)))
print("Validation set:\t\t{}".format(len(data.validation.labels)))

# get class # as an integer instead of one-hot encode
data.test.cls = np.argmax(data.test.labels, axis=1)

# Get some image dimensions
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1        # Grayscale so only one channel
num_classes = 10        # 10 digits, so 10 classes

# A helper function for plotting images taken from the notebook

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true)


# Define some helper functions for creating new variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.0, shape=[length]))


def new_conv_layer(input,               # Previous layer
                   num_input_channels,  # Num channels from previous layer
                   filter_size,         # Width x height of each filter, if 5x5 enter 5 here
                   num_filters,         # Number of filters
                   use_pooling=True):    # Use 2x2 max pooling or not
    # shape determined by Tensorflow API
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create some new weights for the shape above and initialize them randomly
    weights = new_weights(shape=shape)

    # Create one bias for each filter
    biases = new_biases(length=num_filters)

    # Create Tensorflow convolution operation
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],  # first and last stride must always be 1
                         padding='SAME')        # padding: what to do at edge of image

    # Add biases to the reuslts of convolution:
    layer += biases

    # Use pooling if indicated:
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                              ksize=[1, 2, 2, 1],
                              strides=[1,2,2,1],
                              padding='SAME')

    # Then use a RELU to introduce some non-linearity
    layer = tf.nn.relu(layer)

    # ReLU is normally executed before pooling
    # but relu(max_pool(x)) == max_pool(relu(x))
    # So would rather run ReLU on a smaller piece (1x1 as opposed to 2x2)

    # return both layer and filter weights for later use when running the session
    return layer, weights


# Helper function to flatten a layer, i.e. when feeding form a conv layer into a fully connected
def flatten_layer(layer):
    # Get shape of input
    input_shape = layer.get_shape()

    # format of shape should be [num_images, img_height, img_width, num_channels]
    # total # of features is therefore img_height * img_width * num_channels; grab this
    num_features = input_shape[1:4].num_elements()

    # flatten to 2D, leaving the first dimension open
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


# Helper function to make a fully connected layer
def new_fc_layer(input,             # previous layer
                 num_inputs,        # number of inputs from previous layer
                 num_outputs,       # of outputs
                 use_relu=True):    # Use a ReLU or not

    # Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Layer is matrix mult of inputs by weights, plus bias
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# Create some placeholders for the stuff that will be fed in through feed_dict
# inputs
# Using None for first dimension allows arbitrary amt of images to be fed in
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
# convert from one-hot to the clas number
y_true_cls = tf.argmax(y_true, dimension=1)

# Make conv layer 1, takes in image
layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                                            num_input_channels = num_channels,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            use_pooling=True)

# Make conv layer 2, takes in output of layer 1
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels = num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)

# Make the flat layer, takes in output of conv 2
layer_flat, num_features = flatten_layer(layer_conv2)

# Make fully connected layer 1, takes in output of flat layer, output is # of neurons
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu = True)

# Make fully connected layer 2, which takes in 128 things and outputs a vector of 10 (logits)
layer_fc2 = new_fc_layer(input = layer_fc1,
                         num_inputs = fc_size,
                         num_outputs=num_classes,
                         use_relu=False)    # Don't use ReLU on final layer; pass to a softmax

# pass logits into softmax, get predictions out in the form of probabilities
y_pred = tf.nn.softmax(layer_fc2)  # DON'T FEED THIS INTO tf.nn.softmax_cross_entropy_with_logits()

# using probabilities, get most likely class
y_pred_cls = tf.argmax(y_pred, dimension=1)


# Now define the cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

# convert softmax to a single scalar value to calculate and optimize on cost:
cost = tf.reduce_mean(cross_entropy)

# define our optimizer: use SGD, minimize cost
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# The notes use something called AdamOptimizer: check this out


# DEFINE SOME PERFORMANCE MEASURES

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# this creates a vector of trues and falses
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# turn previous vector into a % value; this returns TRAINING accuracy


# Now actually run all this; do some tensorflow stuff
session = tf.Session()
session.run(tf.initialize_all_variables())  # initialize all variables


def optimize(epochs):
    start_time = time.time()
    for i in range(0, epochs):

        # Get a batch
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        # put batch into feed_dict
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run optimizer
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every print_step epochs
        if i % print_step == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            print("Epoch: {0} - Training accuracy: {1}".format(i, acc))
    end_time = time.time()

    print("Time: {} sec".format(end_time - start_time))
    print_test_accuracy()


def print_test_accuracy():

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    print("Calculating test accuracy...")
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))


if __name__ == "__main__":
    main()