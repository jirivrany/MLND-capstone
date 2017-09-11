# coding: utf-8

"""
@author Jiri Vrany
@license MIT

"""

import glob
import numpy as np
import tensorflow as tf
import os
from scipy.signal import wiener
from sklearn import preprocessing


def transform_row(row):
    row = row.replace("[", "")
    row = row.replace("],", "")
    return row


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_label(filename):
    """
    create label from the file name
    """
    keys = {"Sphere": 0, "Vertical": 1, "Horizontal": 2}
    names = filename.split(os.sep)
    names = names[4].split()
    return keys[names[0].strip()]


def sanitize_name(filename):
    filename = filename.replace(".txt", "")
    filename = filename.replace(" ", "_")
    filename = filename.split(os.sep)[-1]
    return filename


def create_files(files, output_dir, testset=False):

    for i, fname in enumerate(files):
        with open(fname) as fdata:
            data = fdata.readlines()
            data = [transform_row(data[0]).split(",") for row in data]
            data = np.array(data)
            data = data.astype(np.float64)
            # filter the noise
            data = wiener(data)
            # normalize data
            data = preprocessing.normalize(data, norm='l2')
            # flip the data
            data_s1 = np.fliplr(data)
            data_s2 = np.flipud(data)

        filename = os.path.join(
            output_dir, sanitize_name(fname) + '.tfrecords')
        filename_s1 = os.path.join(
            output_dir, sanitize_name(fname) + 'split_left.tfrecords')
        filename_s2 = os.path.join(
            output_dir, sanitize_name(fname) + 'split_up.tfrecords')

        if not i % 500:
            print(i, 'writing', filename)

        if testset:    
            pairs = ((filename, data),)
        else:    
            pairs = ((filename, data), (filename_s1, data_s1), (filename_s2, data_s2))
        

        for fn, dn in pairs:
            writer = tf.python_io.TFRecordWriter(fn)
            features = tf.train.Features(feature={
                'height': _int64_feature(100),
                'width': _int64_feature(100),
                'depth': _int64_feature(1),
                'label': _int64_feature(create_label(fname)),
                'image': _bytes_feature(dn.tostring())
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            writer.close()


def create_training():

    files = glob.glob('../data/gravimetrie/random_zeronoise/*.txt')
    output_dir = '../data/gravimetrie/random_tf_normalized'
    create_files(files, output_dir)


def create_validation():

    files = glob.glob('../data/gravimetrie/validation_set/*.txt')
    output_dir = '../data/gravimetrie/validation_set_tf'
    create_files(files, output_dir, True)


if __name__ == "__main__":
    create_validation()
