import tensorflow as tf
import numpy as np


def triplet_loss(anchor, positive, negative, N):
    # anchor p*d, positive p*d, negative p*d
    with tf.variable_scope('triplet_loss'):
        p = tf.shape(anchor)[0]
        pos_dist = tf.norm(tf.subtract(anchor, positive), ord=1)
        neg_dist = tf.norm(tf.subtract(anchor, negative), ord=1)
        basic_loss = tf.subtract(pos_dist, neg_dist)
        loss = tf.reduce_sum(tf.nn.sigmoid(basic_loss))
        loss = loss * tf.math.floor(N/p)
    return loss


def encoder(placeholder_x, bits):
    with tf.variable_scope("encoder"):
        fc1 = tf.layers.dense(placeholder_x,
                              bits,
                              activation=tf.nn.tanh,
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    return fc1


def get_loss(placeholder_x, bits, p):
    # placeholder_x (2+p)*input_dim
    encoded = encoder(placeholder_x, bits)
    anchor = encoded[:, :1]
    anchors = tf.reshape(tf.tile(anchor, p), [p, -1])
    positive = tf.tile(encoded[:, 1:2])
    positives = tf.reshape(tf.tile(positive, p), [p, -1])
    negatives = encoded[:, 2:]
    loss = triplet_loss(anchors, positives, negatives)
    return loss


def train_model(train_x, placeholder_x, bits, learning_rate):
    num_iterations = 100
    loss = get_loss(placeholder_x, bits)
    model_saver = tf.train.Saver(max_to_keep=None)
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    graph_path = './model/'
    path = graph_path + 'model'
    m = np.shape(train_x)[0]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        L = tf.summary.scalar('loss', loss)
        writer = tf.summary.FileWriter(graph_path, sess.graph)
        for n_pass in range(num_iterations):
            p = np.random.randint(m)
            A = train_x[p]
            P =
            N =
            x_batch = np.concatenate([A, P, N])
            feed_dict = {placeholder_x: x_batch}
            _, loss_eval, res = sess.run([opt, loss, L], feed_dict = feed_dict)
            print("Batch {} finished. loss={}".format(n_pass, loss_eval))
            writer.add_summary(res, n_pass)
        model_saver.save(sess=sess, save_path=path)
    return


def test_model():
    return


def main():
    input_dim = 100
    output_dim = 10
    with tf.variable_scope("placeholders"):
        series_var = tf.placeholder(tf.uint8, shape=(None, input_dim), name="series")
    x = np.array([1,2,3,4,5,6])
    train_model(x, series_var, output_dim)
