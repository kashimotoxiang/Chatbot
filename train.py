import tensorflow as tf
from models import model
import time
import os
import cn_hparams
import datetime

FLAGS = tf.flags.FLAGS
TIMESTAMP = int(time.time())

if FLAGS.RNN_CNN_MaxPooling_model_dir:
    MODEL_DIR = FLAGS.RNN_CNN_MaxPooling_model_dir
else:
    MODEL_DIR = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))
TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(os.path.join(
    FLAGS.input_dir, "validation.tfrecords"))
tf.logging.set_verbosity(FLAGS.loglevel)

def main():
    hparams = cn_hparams.create_hparams()

    with tf.Graph().as_default():
      with tf.device("/gpu:1"):
          with tf.Session() as sess:

            training_Model=model.Model(hparams,
                                      model_fun=model.RNN,
                                      RNNInit=tf.nn.rnn_cell.LSTMCell,
                                      is_bidirection=False)
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-1)
            # optimizer = tf.train.GradientDescentOptimizer(1e-2)
            grads_and_vars = optimizer.compute_gradients(training_Model.mean_loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-1)
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", training_Model.mean_loss)
            acc_summary = tf.summary.scalar("accuracy", training_Model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary,  grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(context_batch,
                           context_len_bacth,
                           utterance_batch,
                           utterance_len_batch,
                           targets_batch):
                """
                A single training step
                """
                feed_dict = {
                    training_Model.context_idx:context_batch,
                    training_Model.context_len_idx:context_len_bacth,
                    training_Model.utterance_idx:utterance_batch,
                    training_Model.utterance_len_idx:utterance_len_batch,
                    training_Model.targets: targets_batch
                }
                _, step, summaries, loss = sess.run(
                    [train_op, global_step, train_summary_op, training_Model.mean_loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss))
                train_summary_writer.add_summary(summaries, step)

            def _recall_at_k(predictions, labels, k):
                positive_labels = tf.where(tf.equal(labels, 1))[:, 1]
                TP = tf.nn.in_top_k(predictions, targets=positive_labels, k=k)
                TP = tf.reduce_sum(tf.cast(TP, tf.int32))
                T = tf.to_int32((tf.shape(positive_labels)[0]))
                return TP / T

            def devstep(context_batch,
                           context_len_bacth,
                           utterance_batch,
                           utterance_len_batch,
                           targets_batch):
                feed_dict = {
                    training_Model.context_idx: context_batch,
                    training_Model.context_len_idx: context_len_bacth,
                    training_Model.utterance_idx: utterance_batch,
                    training_Model.utterance_len_idx: utterance_len_batch,
                    training_Model.targets:targets_batch
                }
                probs = sess.run(
                    training_Model.probs,
                    feed_dict=feed_dict
                )
                # 计算精度
                split_probs = tf.split(probs, 10, 0)
                shaped_probs = tf.concat(split_probs, 1)
                prediction = tf.argmax(shaped_probs, 1)
                correct = tf.cast(tf.equal(prediction, 0), tf.float32)
                accuracy = tf.reduce_mean(correct)
                # 计算top k
                labels = tf.reshape(targets_batch, [-1, 1 + hparams.distraction_num])
                recall_at_1 = _recall_at_k(shaped_probs, labels, 1)
                recall_at_2 = _recall_at_k(shaped_probs, labels, 2)
                recall_at_5 = _recall_at_k(shaped_probs, labels, 5)
                recall_at_8 = _recall_at_k(shaped_probs, labels, 8)
                recall_at_10 = _recall_at_k(shaped_probs, labels, 10)

