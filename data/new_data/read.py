import tensorflow as tf
import numpy as np


def decode(serialized_example):
    features = {"context": tf.FixedLenFeature(160, tf.int64),
                "context_len": tf.FixedLenFeature(1, tf.int64),
                "utterance": tf.FixedLenFeature(160, tf.int64),
                "utterance_len": tf.FixedLenFeature(1, tf.int64),
                "label": tf.FixedLenFeature(1, tf.int64)
                }
    parsed_features = tf.parse_single_example(serialized_example, features)
    # context = parsed_features["context"]
    return parsed_features["context"], parsed_features["context_len"], parsed_features["utterance"], parsed_features[
        "utterance_len"], parsed_features["label"]


filename = ["train.tfrecords"]
dataset = tf.data.TFRecordDataset(filename).map(decode)
iterator = dataset.make_one_shot_iterator()
context_i, context_len_i, utterance_i, utterance_len_i, label_i = iterator.get_next()

context = []
context_len = []
utterance = []
utterance_len = []
label = []

with tf.Session() as sess:
    try:
        while True:
            context.append(sess.run(context_i, ))
            context_len.append(sess.run(context_len_i, ))
            utterance.append(sess.run(utterance_i, ))
            utterance_len.append(sess.run(utterance_len_i, ))
            label.append(sess.run(label_i, ))
    except tf.errors.OutOfRangeError:
        pass

context = np.array(context)
context_len = np.array(context_len)
utterance = np.array(utterance)
utterance_len = np.array(utterance_len)
label = np.array(label)
pass
