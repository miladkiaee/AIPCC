# suppress the possible cpu architecture warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pylab
#import tensorflow core
import tensorflow as tf

tf.reset_default_graph()

# config
N = 50
batch_size = 100
learning_rate = 0.5
training_epochs = 5
logs_path = "/home/milad/PycharmProjects/sync_coin_database/logs/"

##################################################################
# beginning of computational graph

# queue holder for file names
filename_queue = tf.train.string_input_producer(["ada_history.csv"])
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [tf.constant([], dtype=tf.float32),
                   tf.constant([], dtype=tf.float32),
                   tf.constant([], dtype=tf.float32),
                   tf.constant([], dtype=tf.float32),
                   tf.constant([], dtype=tf.float32),
                   tf.constant([], dtype=tf.float32),
                   tf.constant([], dtype=tf.float32),
                   tf.constant([], dtype=tf.float32)]

# setting column tensor nodes
col1, col2, col3, col4, col5 , col6, col7, col8 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8])

####################################################################
# training variables
with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    #batches are for training
    x = tf.placeholder(col6, shape=[None, N], name="x-input")
    # target 10 output classes
    # clasifiers
    y_ = tf.placeholder(col6, shape=[None, 10], name="y-input")

####################################################################
# define summaries
# model parameters will change during training so we use tf.Variable
with tf.name_scope("weights"):
    W = tf.Variable(tf.zeros([N, 10]))

# bias
with tf.name_scope("biases"):
    b = tf.Variable(tf.zeros([10]))

# implement model
with tf.name_scope("softmax"):
    # y is our prediction
    y = tf.nn.softmax(tf.matmul(x, W) + b)

# specify cost function
with tf.name_scope('cross_entropy'):
    # this is our cost
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# specify optimizer
with tf.name_scope('train'):
    # optimizer is an "operation" which we can execute in a session
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create a summary for our cost and accuracy
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)

# merge all summaries into a single "operation" which we can execute in a session
summary_op = tf.summary.merge_all();

# end of computational graph
####################################################################
# start of session
with tf.Session() as sess:
  # Start populating the filename queue.
  # training the core
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        # Retrieve a single instance:
        example, label = sess.run([features, col5])
        print(example)

    coord.request_stop()
    coord.join(threads)
    coord.join(threads)


# end of code
###################################################################