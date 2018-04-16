import tensorflow as tf
import models
import time
import os
import hparams as hp
from training_method import Training

FLAGS = tf.flags.FLAGS
MODEL_DIR = FLAGS.RUNS
CHECKPOINT_DIR = os.path.abspath(os.path.join(MODEL_DIR, "checkpoints"))

TIMESTAMP = int(time.time())
tf.logging.set_verbosity(FLAGS.loglevel)


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
    optimizer = tf.train.AdamOptimizer(hp.learning_rate)
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
    training = Training(sess=sess,
                        model=training_Model,
                        train_op=train_op,
                        global_step=global_step,
                        hparams=hparams,
                        _trarning_path=FLAGS.TRAIN_FILE,
                        _validation_path=FLAGS.VALIDATION_FILE,
                        _test_path=FLAGS.TEST_FILE)

    for _ in range(FLAGS.num_epochs):

        # training
        feature_map, targets_batch=training.data_loader.load_train_data()
        context_batch = feature_map['context']
        context_len_bacth = feature_map['context_len']
        utterance_batch = feature_map['utterance']
        utterance_len_batch = feature_map['utterance_len']

        training.train_step(context_batch, context_len_bacth,
                            utterance_batch, utterance_len_batch, targets_batch)
        current_step = tf.train.global_step(sess, global_step)

        # eval
        if current_step % FLAGS.evaluate_every == 0:
            context_batch, context_len_bacth, utterance_batch, utterance_len_batch, targets_batch = training.data_loader.load_eval_data(
                FLAGS.batch_size, FLAGS.max_len)
            accuracy, recall_at_1,recall_at_2,recall_at_5,recall_at_8,recall_at_10=\
                training.eval_step(context_batch, context_len_bacth,
                               utterance_batch, utterance_len_batch, targets_batch)
            print('step {}:Accuracy={},recall_at_1={},'
                  'recall_at_2={},recall_at_5={}，'
                  'recall_at_8={}，recall_at_10={}'.format(current_step, accuracy,
                                                          recall_at_1, recall_at_2, recall_at_5,
                                                          recall_at_8, recall_at_10))

        # save checkpoint
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, CHECKPOINT_DIR,
                              global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
