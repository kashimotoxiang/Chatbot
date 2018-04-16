import tensorflow as tf
import models
import time
import os
import datetime
import data_loader as dl
import numpy as np


class Training():
    sess = None
    model = None
    global_step = None
    train_op = None
    hparams = None
    data_loader = None

    def __init__(self, sess, model, train_op, global_step, hparams, _trarning_path, _validation_path, _test_path):
        self.sess = sess
        self.model = model
        self.global_step = global_step
        self.train_op = train_op
        self.hparams = hparams
        self.data_loader = dl.DataLoader(
            sess, _trarning_path, _validation_path, _test_path)

    def train_step(self,
                   context_batch,
                   context_len_bacth,
                   utterance_batch,
                   utterance_len_batch,
                   targets_batch):
        """
        A single training step
        """
        feed_dict = {
            self.model.context_idx: context_batch,
            self.model.context_len: context_len_bacth,
            self.model.utterance_idx: utterance_batch,
            self.model.utterance_len: utterance_len_batch,
            self.model.targets: targets_batch
        }
        _, step, loss = self.sess.run(
            [self.train_op, self.global_step, self.model.mean_loss],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}".format(time_str, step, loss))

    def _recall_at_k(self, predictions, labels, k):
        # positive_labels = tf.where(tf.equal(labels, 1))[:, 1]
        # TP = tf.nn.in_top_k(predictions, targets=positive_labels, k=k)
        # TP = tf.reduce_sum(tf.cast(TP, tf.int32))
        # T = tf.to_int32((tf.shape(positive_labels)[0]))
        # return TP / T
        # where返回两个array，这样可以得到坐标tuple
        positive_labels = list(zip(*np.where(labels == 1)))
        top_k = np.argsort(-predictions, axis=1)[:, 0:k]
        batch_size = np.shape(labels)[0]
        idx = np.arange(0, batch_size).reshape(-1, 1)
        # 重复idx k次
        idx = np.tile(idx, (1, k))
        idx = idx.flatten()
        top_k = top_k.flatten()
        top_k = list(zip(idx, top_k))

        true_positive_num = len(set(positive_labels) & set(top_k))
        return true_positive_num / len(positive_labels)

    def eval_step(self, context_batch,
                  context_len_bacth,
                  utterance_batch,
                  utterance_len_batch,
                  targets_batch):
        feed_dict = {
            self.model.context_idx: context_batch,
            self.model.context_len: context_len_bacth,
            self.model.utterance_idx: utterance_batch,
            self.model.utterance_len: utterance_len_batch,
            self.model.targets: targets_batch
        }
        probs = self.sess.run(
            self.model.probs,
            feed_dict=feed_dict
        )
        # 计算精度
        # split_probs = tf.split(probs, 10, 0)
        # shaped_probs = tf.concat(split_probs, 1)
        # prediction = tf.argmax(shaped_probs, 1)
        # correct = tf.cast(tf.equal(prediction, 0), tf.float32)
        # accuracy = tf.reduce_mean(correct)
        shaped_probs = np.reshape(
            probs, [-1, 1 + self.hparams.distraction_num])
        prediction = np.argmax(shaped_probs, 1)
        correct = np.equal(prediction, 0)
        accuracy = np.mean(correct.astype(int))
        # 计算top k
        labels = tf.reshape(
            targets_batch, [-1, 1 + self.hparams.distraction_num])
        recall_at_1 = self._recall_at_k(shaped_probs, labels, 1)
        recall_at_2 = self._recall_at_k(shaped_probs, labels, 2)
        recall_at_5 = self._recall_at_k(shaped_probs, labels, 5)
        recall_at_8 = self._recall_at_k(shaped_probs, labels, 8)
        recall_at_10 = self._recall_at_k(shaped_probs, labels, 10)
