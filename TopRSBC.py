import tensorflow as tf
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split


def data_pre(data):
    mean = tf.reduce_mean(data, axis=1)
    data_p = data - mean
    # print(data.shape, mean.shape, data_p.shape)
    return data_p


def triplet_loss(anchor, positive, negative, n):
    # anchor p*d, positive p*d, negative p*d
    with tf.variable_scope('triplet_loss'):
        p = anchor.shape[1]
        # print("p{}".format(p))
        pos = anchor - positive
        neg = anchor - negative
        print("pos{}".format(pos.shape))
        pos_dist = tf.norm(pos, ord=1, axis=2)
        neg_dist = tf.norm(neg, ord=1, axis=2)
        basic_loss = pos_dist - neg_dist
        sigmoid_basic_loss = tf.nn.sigmoid(basic_loss)
        print(anchor.shape, pos_dist.shape, basic_loss.shape, sigmoid_basic_loss.shape, "loss")
        sum_loss = tf.reduce_sum(sigmoid_basic_loss)
        loss = tf.reduce_mean(tf.math.log(1 + tf.math.floor(n / p) * sum_loss))
        print(loss.shape)
    return loss


def encoder(placeholder_x, bits):
    with tf.variable_scope("encoder"):
        fc1 = tf.layers.dense(placeholder_x,
                              bits,
                              use_bias=False,
                              activation=tf.nn.tanh,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    # print(placeholder_x.shape, fc1.shape)
    return fc1


def get_loss(placeholder_x, bits, p, n):
    # placeholder_x (2+p)*input_dim
    # data_p = data_pre(placeholder_x)  # pre
    encoded = encoder(placeholder_x, bits)
    anchor = encoded[:, 0, :]
    print("anchor{}".format(anchor.shape))
    dim = anchor.shape[1]
    # print(dim, "#")
    anchors = tf.reshape(tf.tile(anchor, [1, p]), [-1, p, dim])
    positive = encoded[:, 1, :]
    positives = tf.reshape(tf.tile(positive, [1, p]), [-1, p, dim])
    negatives = encoded[:, 2:, :]
    print(anchors.shape, negatives.shape, positives.shape)
    loss = triplet_loss(anchors, positives, negatives, n) + tf.losses.get_regularization_loss()
    return loss, anchor


def train_model(train_x, placeholder_x, placeholder_n, bits, learning_rate, Num_p):
    # num = np.shape(train_x)[0]  # sample number
    max_class = int(max(train_x[:, -1])) + 1  # 0 1 2...
    print("max{}".format(max_class))
    num_iterations = 1000
    loss, _ = get_loss(placeholder_x, bits, Num_p, placeholder_n)
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    graph_path = './model/'
    path = graph_path + 'model'
    model_saver = tf.train.Saver(max_to_keep=None)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        L = tf.summary.scalar('loss', loss)
        writer = tf.summary.FileWriter(graph_path, sess.graph)
        for n_pass in range(num_iterations):
            anchor_class = np.random.randint(max_class)
            positives = train_x[train_x[:, -1] == anchor_class]
            # print(positives.shape, anchor_class)
            a, b = np.random.randint(len(positives), size=2)
            A = positives[a:a+1]
            P = positives[b:b+1]
            negatives = train_x[train_x[:, -1] != anchor_class]
            Num_n = len(negatives)
            slices = np.random.randint(Num_n, size=Num_p)
            N = negatives[slices]
            # print(A.shape, P.shape, N.shape, Num_n)
            x_batch = np.concatenate((A, P, N), axis=0)[:, :-1]
            x_batch = x_batch[np.newaxis, :]
            # print(x_batch.shape)
            feed_dict = {placeholder_x: x_batch, placeholder_n: np.array([Num_n])[np.newaxis, :]}
            _, loss_eval, result = sess.run([opt, loss, L], feed_dict=feed_dict)
            print("Batch {} finished. loss={}".format(n_pass, loss_eval))
            writer.add_summary(result, n_pass)
        model_saver.save(sess=sess, save_path=path)
        writer.close()
    return


def test_model(test_x, placeholder_x, bits, placeholder_n, Num_p):
    _, anchor = get_loss(placeholder_x, bits, Num_p, placeholder_n)
    output = []
    saver = tf.train.Saver()
    with tf.Session() as sess:
        for i in range(len(test_x)):
            path = "./model/"
            saver.restore(sess, tf.train.latest_checkpoint(path))
            input_x = np.reshape(np.tile(test_x[i], [1, Num_p+2]), (1, Num_p+2, -1))
            feed_dict = {placeholder_x: input_x, placeholder_n: np.array([1])[np.newaxis, :]}
            encoded = sess.run(anchor, feed_dict=feed_dict)
            output.append(encoded[0])
    output = np.array(output)
    return


def main():
    data = sio.loadmat("./datasets/eeg.mat")["data"]
    data[np.isnan(data)] = 0
    data = data[:, :-1] - np.mean(data[:, :-1], axis=0)
    train_x, test_x = train_test_split(data, test_size=0.2)
    input_dim = np.shape(train_x)[1] - 1  # features+label
    output_dim = 64  # bits
    Num_p = 1800
    learning_rate = 0.01
    with tf.variable_scope("placeholders"):
        series_var = tf.placeholder(tf.float64, shape=(None, Num_p+2, input_dim), name="series")
        n_var = tf.placeholder(tf.int32, shape=(None, 1), name="N")
    train_model(train_x, series_var, n_var, output_dim, learning_rate, Num_p)


if __name__ == "__main__":
    main()
