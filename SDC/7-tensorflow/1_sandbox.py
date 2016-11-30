import tensorflow as tf

# Create a Tensorflow object named tensor
hello_constant = tf.constant('Hello World')

# Create a placeholder Tensor that takes a value passed in from feed_dict parameter
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run([x, y, z], feed_dict={x: 'Hello World', y: 123, z: 45.67})
    print(output)


