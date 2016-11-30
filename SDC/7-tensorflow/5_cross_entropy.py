# Solution is available in the other "solution.py" tab
import tensorflow as tf


def run():
    softmax_data = [0.7, 0.2, 0.1]
    one_hot_encod_label = [1.0, 0.0, 0.0]

    softmax = tf.placeholder(tf.float32)
    one_hot_encod = tf.placeholder(tf.float32)

    # don't forget that the cross_entropy is negative!!!!
    cross_entropy = -tf.reduce_sum(tf.mul(one_hot_encod,tf.log(softmax_data)))

    with tf.Session() as sess:
        output = sess.run(cross_entropy, feed_dict={softmax: softmax_data,
                                                one_hot_encod: one_hot_encod_label})

    print output
    return output


def main():
    run()


if __name__ == "__main__":
    main()