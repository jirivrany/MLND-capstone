"""
@author Jiri Vrany
@license MIT

Final CNN clasifier. Implemeted with TensorFlow and Pandas.

Based on TensorFlow CNN tutorials 

"""

import tensorflow as tf
import glob
import sys
from math import sqrt
import random
import pandas as pd

TRDATA = 72000
DEPTH = 1
NrCLASS = 3
BATCH_SIZE = 64
EPOCHS = 2
STEPS = TRDATA // BATCH_SIZE

#basic settings
directory = "../data/gravimetrie/random_tf_normalized/*.tfrecords"
filenames = glob.glob(directory)
print("FILES: ", len(filenames))
random.shuffle(filenames)

validation_directory = "../data/gravimetrie/validation_set_tf/*.tfrecords"
val_filenames = glob.glob(validation_directory)

training_queue = tf.train.string_input_producer(
    filenames[:TRDATA], shuffle=True, num_epochs=EPOCHS)
test_queue = tf.train.string_input_producer(
    filenames[TRDATA:], shuffle=True, num_epochs=EPOCHS)
valid_queue = tf.train.string_input_producer(
    val_filenames, shuffle=False, num_epochs=1)


def read_and_decode(filename_queue):
    """
    function for reading TF records samples from the filename_queue
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)

        })
    image = tf.decode_raw(features['image'], tf.float64)
    label = tf.cast(features['label'], tf.int32)
    image = tf.reshape(image, [100 * 100])
    label = tf.stack(tf.one_hot(label, NrCLASS))
    
    return image, label

# functions to init small positive weights and biases


def weight_variable(shape, stddev=0.01):
    """
    helper for the weight variable 
    @returns tensor flow variable of given shape
    """
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, stddev=0.01):
    """
    helper for the bias variable 
    @returns tensor flow variable of given shape
    """
    initial = tf.constant(stddev, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    helper for convolutional layer 
    @returns tensor flow conv2d layer 
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    helper for pooling layer 
    @returns tensor flow pooling layer 
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# the main CNN code
with tf.Session() as sess:
    # get the data
    image, label = read_and_decode(training_queue)
    test_image, test_label = read_and_decode(test_queue)
    valid_image, valid_label = read_and_decode(valid_queue)

    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=BATCH_SIZE,
        capacity=2000,
        min_after_dequeue=1000)

    test_image_batch, test_label_batch = tf.train.shuffle_batch(
        [test_image, test_label], batch_size=BATCH_SIZE,
        capacity=2000,
        min_after_dequeue=1000)

    valid_image_batch, valid_label_batch = tf.train.shuffle_batch(
        [valid_image, valid_label], batch_size=100,
        capacity=2000,
        min_after_dequeue=1000)

    # CNN implementation
    x = tf.placeholder(tf.float32, [None, 100 * 100])
    y_ = tf.placeholder(tf.float32, [None, NrCLASS])

    filters1 = 128
    filters2 = 256
    nNeuronsfc1 = 1024
    
    # input
    x_image = tf.reshape(x, [-1, 100, 100, 1])

    # conv layer 1
    W_conv11 = weight_variable([7, 7, 1, filters1], stddev=0.001)
    b_conv11 = bias_variable([filters1])
    h_conv11 = tf.nn.relu(conv2d(x_image, W_conv11) + b_conv11)

    h_pool1 = max_pool_2x2(h_conv11)

    # conv layer 2
    W_conv2 = weight_variable([5, 5, filters1, filters2], stddev=0.001)
    b_conv2 = bias_variable([filters2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    
    # fully connected layer 1
    W_fc1 = weight_variable([25 * 25 * filters2, nNeuronsfc1], stddev=0.001)
    b_fc1 = bias_variable([nNeuronsfc1])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 25 * filters2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout on fc1
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variable([nNeuronsfc1, NrCLASS])
    b_fc2 = bias_variable([NrCLASS])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # training

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                                  tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_time = []

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for e in range(EPOCHS):
        for i in range(TRDATA // BATCH_SIZE):
            batch_xs, batch_ys = sess.run([image_batch, label_batch])

            train_step.run(
                feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.6})
            if (i + 1) % 10 == 0:  # then perform validation
                # get a validation batch
                vbatch_xs, vbatch_ys = sess.run(
                    [test_image_batch, test_label_batch])
                train_accuracy = accuracy.eval(feed_dict={
                    x: vbatch_xs, y_: vbatch_ys, keep_prob: 1.0})
                print("epoch %d, step %d, training accuracy %g" %
                      (e + 1, i + 1, train_accuracy))

                acc_time.append(train_accuracy)

    valid_batch_xs, valid_batch_ys = sess.run(
        [valid_image_batch, valid_label_batch])
    final_dict = {x: valid_batch_xs, y_: valid_batch_ys, keep_prob: 1.0}

    final_accuracy = accuracy.eval(feed_dict=final_dict)
    print("final test accuracy %g" % (final_accuracy))
    
    expected = []
    predicted = []

    for st in range(10):
        test_labels = tf.argmax(y, axis=1)
        true_labels = tf.argmax(y_, axis=1)
        test_pred = sess.run(test_labels, feed_dict=final_dict)
        true_pred = sess.run(true_labels, feed_dict=final_dict)
        expected.extend(true_pred)
        predicted.extend(test_pred)
    
    print(expected)    
    print(predicted)

    print("writing output")   
    #Submision
    submission = pd.DataFrame(data={'expected':expected, 'predicted':predicted})
    submission.to_csv('final_cnn64_classif.csv', index=False)
    submission.tail()  

    mychart = pd.DataFrame(data={'accuracy': acc_time})
    mychart.to_csv('final_cnn64_accuracy.csv', index=False)
    mychart.tail() 

    coord.request_stop()
    coord.join(threads)

    