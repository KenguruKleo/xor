import tensorflow as tf
import matplotlib.pyplot as plt


def train():
    # Parameters
    learning_rate = 0.03
    epochs_count = 100
    show_steps = 1000

    X = tf.placeholder(tf.float32, [4, 2])

    W1 = tf.Variable(tf.random_uniform([2, 2], -1, 1))
    W2 = tf.Variable(tf.random_uniform([2, 1], -1, 1))

    b1 = tf.Variable(tf.zeros([2]))
    b2 = tf.Variable(tf.zeros([1]))

    layer1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    Y = tf.nn.sigmoid(tf.matmul(layer1, W2) + b2)

    Y_ = tf.placeholder(tf.float32, [4, 1])

    XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    XOR_Y = [[0], [1], [1], [0]]

    accuracy = tf.reduce_mean(-(Y_ * tf.log(Y) + (1 - Y_) * tf.log(1.0 - Y)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(accuracy)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    tf.summary.FileWriter("./logs", sess.graph)
    sess.run(init)

    accuracy_results = []

    epochs = range(epochs_count)
    for i in epochs:
        for j in range(show_steps):
            sess.run(train_step, feed_dict={X: XOR_X, Y_: XOR_Y})

        curr_accuracy = sess.run(accuracy, feed_dict={X: XOR_X, Y_: XOR_Y})
        print(i, curr_accuracy)
        # print('Y', sess.run(Y, feed_dict={X: XOR_X, Y_: XOR_Y}))
        accuracy_results.append(curr_accuracy)

    print('accuracy', sess.run(accuracy, feed_dict={X: XOR_X, Y_: XOR_Y}))
    print('Y', sess.run(Y, feed_dict={X: XOR_X, Y_: XOR_Y}))
    print('W1', sess.run(W1))
    print('W2', sess.run(W2))
    print('b1', sess.run(b1))
    print('b2', sess.run(b2))

    plt.plot(epochs, accuracy_results, 'r')
    plt.legend(['accuracy'])
    plt.show()
