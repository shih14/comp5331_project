import tensorflow as tf
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split


def data_pre(data):
    data_p = data - tf.reduce_mean(data, 0)
    return data_p


def triplet_loss(anchor, positive, negative, n):
    # anchor p*d, positive p*d, negative p*d
    with tf.variable_scope('triplet_loss'):
        p = tf.shape(anchor)[0]
        pos_dist = tf.norm(tf.subtract(anchor, positive), ord=1)
        neg_dist = tf.norm(tf.subtract(anchor, negative), ord=1)
        basic_loss = tf.subtract(pos_dist, neg_dist)
        sum_loss = tf.reduce_sum(tf.nn.sigmoid(basic_loss))
        loss = tf.math.log(1 + tf.math.floor(n / p) * sum_loss)
    return loss


def encoder(placeholder_x, bits):
    with tf.variable_scope("encoder"):
        fc1 = tf.layers.dense(placeholder_x,
                              bits,
                              activation=tf.nn.tanh,
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    return fc1


def get_loss(placeholder_x, bits, p, n):
    # placeholder_x (2+p)*input_dim
    data_p = data_pre(placeholder_x)  # pre
    encoded = encoder(data_p, bits)
    anchor = encoded[:, :1]
    anchors = tf.reshape(tf.tile(anchor, p), [p, -1])
    positive = tf.tile(encoded[:, 1:2])
    positives = tf.reshape(tf.tile(positive, p), [p, -1])
    negatives = encoded[:, 2:]
    loss = triplet_loss(anchors, positives, negatives, n)
    return loss


def train_model(train_x, placeholder_x, placeholder_n, bits, learning_rate):
    num_iterations = 100
    loss = get_loss(placeholder_x, bits)
    model_saver = tf.train.Saver(max_to_keep=None)
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    graph_path = './model/'
    path = graph_path + 'model'
    num = np.shape(train_x)[0]  #sample number
    Num_p = 200
    max_class = max(train_x[:, -1]) + 1   # 0 1 2...
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        L = tf.summary.scalar('loss', loss)
        writer = tf.summary.FileWriter(graph_path, sess.graph)
        for n_pass in range(num_iterations):
            anchor_class = np.random.randint(max_class)
            positives = train_x[train_x[:, -1] == anchor_class]
            a, b = np.random.randint(len(positives), size=2)
            A = positives[a]
            P = positives[b]
            negatives = train_x[train_x[:, -1] != anchor_class]
            Num_n = len(negatives)
            slices = np.random.randint(Num_n, size=Num_p)
            N = negatives[slices]
            x_batch = np.concatenate([A, P, N])[:, :-1]
            feed_dict = {placeholder_x: x_batch, placeholder_n: Num_n}
            _, loss_eval, res = sess.run([opt, loss, L], feed_dict=feed_dict)
            print("Batch {} finished. loss={}".format(n_pass, loss_eval))
            writer.add_summary(res, n_pass)
        model_saver.save(sess=sess, save_path=path)
    return


def test_model(test_x, placeholder_x, bits):
    return


def main():
    data = sio.loadmat("./datasets/pamap.mat")["data"]
    train_x, test_x = train_test_split(data, test_size=0.2)
    input_dim = np.shape(train_x)[1] - 1
    output_dim = 10 # bits
    learning_rate = 0.1
    with tf.variable_scope("placeholders"):
        series_var = tf.placeholder(tf.uint8, shape=(None, input_dim), name="series")
        n_var = tf.placeholder(tf.uint8, shape=(None, 1), name="N")
    train_model(train_x, series_var, n_var, output_dim, learning_rate)
