import tensorflow as tf
import models
import time
import os
import hparams as hp
import datetime
from training_method import Training

FLAGS = tf.flags.FLAGS
TIMESTAMP = int(time.time())

MODEL_DIR = FLAGS.RNN_CNN_MaxPooling_model_dir

tf.logging.set_verbosity(FLAGS.loglevel)

CHECKPOINT_DIR = os.path.abspath(
    os.path.join(MODEL_DIR, "checkpoints"))


def main():
    hparams = hp.create_hparams()
    # with tf.Graph().as_default():
    with tf.Session() as sess:
        ##########################
        # Define Training model
        training_Model = models.Model(hparams,
                                      model_fun=models.RNN,
                                      RNNInit=tf.nn.rnn_cell.LSTMCell,
                                      is_bidirection=False)

        ##########################
        # Define Training procedure
        global_step = tf.Variable(
            0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-1)
        # optimizer = tf.train.GradientDescentOptimizer(1e-2)
        grads_and_vars = optimizer.compute_gradients(
            training_Model.mean_loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        ##########################
        # create checkpoint dir it if not exist
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        saver = tf.train.Saver(tf.all_variables())

        ##########################
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        ##########################
        # initializing training class
        training = Training(sess, training_Model,
                            train_op, global_step, hparams)

        for _ in range(FLAGS.num_epoch):

            context_batch, context_len_bacth, utterance_batch, utterance_len_batch, targets_batch = training.load_train_data(
                FLAGS.batch_size, FLAGS.max_len)
            training.train_step(context_batch, context_len_bacth,
                                utterance_batch, utterance_len_batch, targets_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                training.eval_step()
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, CHECKPOINT_DIR,
                                  global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
