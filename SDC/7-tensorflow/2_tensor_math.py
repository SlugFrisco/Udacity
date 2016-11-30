import tensorflow as tf

def run():
    x = tf.constant(10)
    y = tf.constant(2)
    z = tf.sub(tf.div(x,y), tf.constant(1))

    with tf.Session() as sess:
        output = sess.run(z)
        print(output)

    return output


def main():
    run()


if __name__ == "__main__":
    main()