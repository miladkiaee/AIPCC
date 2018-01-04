# suppress the possible cpu architecture warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#import tensorflow core
import tensorflow as tf

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

# end of computational graph
####################################################################
# start of session
with tf.Session() as sess:
  # Start populating the filename queue.
  # training the core
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Retrieve a single instance:
    print(sess.run([features, col6]))

    coord.request_stop()
    coord.join(threads)
    coord.join(threads)

# end of code
###################################################################