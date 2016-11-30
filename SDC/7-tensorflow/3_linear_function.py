# Solution is available in the other "solution.py" tab
import tensorflow as tf


def run():
    output = None
    x_data = [[1.0, 2.0], [2.5, 6.3]]
    test_weights = [[-0.3545495, -0.17928936], [-0.63093454, 0.74906588]]
    # class_size is # of labels
    class_size = 2

    x = tf.placeholder(tf.float32)
    # making weights and biases tf.Variable allows them to be modified over time
    # via SGD
    ### TODO: understand these dimensions better
    weights = tf.Variable(test_weights)
    biases = tf.Variable(tf.zeros([class_size]))

    logits = tf.add(tf.matmul(weights, x), biases)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        output = sess.run(logits, feed_dict={x: x_data})

    print(output)
    return output


def main():
    run()


if __name__ == "__main__":
    main()