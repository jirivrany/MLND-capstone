"""
@author Jiri Vrany
@license MIT

based on TensorFlow CNN tutorial
"""

import tensorflow as tf
import glob
import sys
import pandas as pd


directory = "../data/gravimetrie/random_tf_normalized/*.tfrecords"
filenames = glob.glob(directory)

TRDATA = 72000

training_queue = tf.train.string_input_producer(filenames[:TRDATA], shuffle=True)
test_queue = tf.train.string_input_producer(filenames[TRDATA:], shuffle=True)

validation_directory = "../data/gravimetrie/validation_set_tf/*.tfrecords"
val_filenames = glob.glob(validation_directory)
valid_queue = tf.train.string_input_producer(val_filenames, shuffle=False)


WIDTH = 100
HEIGHT = 100
DEPTH = 1
NrCLASS = 3

def read_and_decode(filename_queue):
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


def weight_variable(shape, stddev=0.01):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, stddev=0.01):
    initial = tf.constant(stddev, shape=shape)
    return tf.Variable(initial)


with tf.Session() as sess:
    image, label = read_and_decode(training_queue)
    test_image, test_label = read_and_decode(test_queue)
    valid_image, valid_label = read_and_decode(valid_queue)

    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=100,
        capacity=2000,
        min_after_dequeue=1000)

    test_image_batch, test_label_batch = tf.train.shuffle_batch(
        [test_image, test_label], batch_size=100,
        capacity=2000,
        min_after_dequeue=1000)


    valid_image_batch, valid_label_batch = tf.train.shuffle_batch(
        [valid_image, valid_label], batch_size=100,
        capacity=2000,
        min_after_dequeue=1000)

    #Simple model
    x = tf.placeholder(tf.float32, [None, WIDTH*HEIGHT])
    y_ = tf.placeholder(tf.float32, [None, NrCLASS])
    W = weight_variable([WIDTH*HEIGHT, NrCLASS])
    b = bias_variable([NrCLASS])
    y = tf.nn.softmax(tf.matmul(x, W) + b)


    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc_time = []


    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(7200):
        batch_xs, batch_ys = sess.run([image_batch, label_batch])
        
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
        if (i+1)%20 == 0: # then perform validation 
          # get a validation batch
          vbatch_xs, vbatch_ys = sess.run([test_image_batch, test_label_batch])
          train_accuracy = accuracy.eval(feed_dict={x:vbatch_xs, y_: vbatch_ys})
          print("step %d, training accuracy %g"%(i+1, train_accuracy))
          acc_time.append(train_accuracy)
    
    valid_batch_xs, valid_batch_ys = sess.run(
        [valid_image_batch, valid_label_batch])
    final_dict = {x: valid_batch_xs, y_: valid_batch_ys}

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
    submission.to_csv('simple_softmax_classif.csv', index=False)
    submission.tail()

    mychart = pd.DataFrame(data={'accuracy': acc_time})
    mychart.to_csv('simple_softmax_accuracy.csv', index=False)
    mychart.tail()  

    coord.request_stop()
    coord.join(threads)
