import tensorflow as tf

# Output depth
k_output = 64

# Image Properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_width, image_height, color_channels])

# Weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_width, filter_size_height, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
# Explanation:
# The code below uses the tf.nn.conv2d() function to compute the convolution with weight as the filter
# and [1, 2, 2, 1] for the strides.
# TensorFlow uses a strcd desktop/ide for each input dimension:
# [batch, input_height, input_width, input_channels].
# we play with stride for input_height and input_width,
# but leave stride for 'batch' and 'input_channels' at 1
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)

# once original image has been squeezed into a high-depth, low width and low height representation
# put the whole thing through a regular neural layer
# then to a classifier

