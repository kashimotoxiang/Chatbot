import os
import time
import tensorflow as tf
import cn_model
import cn_hparams
import cn_metrics
import cn_inputs
from models.model import dual_encoder_model
from tensorflow.contrib.learn import Estimator
from models import model
import os
# os.environ["CUDA VISIBLE DEVICES"] = "3"


FLAGS = tf.flags.FLAGS

TIMESTAMP = int(time.time())

MODEL_DIR = FLAGS.RUNS
TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(os.path.join(
    FLAGS.input_dir, "validation.tfrecords"))

tf.logging.set_verbosity(FLAGS.loglevel)


def main(unused_argv):
    hparams = cn_hparams.create_hparams()
    # model_fun=[2,3,4,5],30
    model_fn = cn_model.create_model_fn(
        hparams,
        model_impl=dual_encoder_model,
        model_fun=model.RNN,
        RNNInit=tf.nn.rnn_cell.LSTMCell,
        is_bidirection=True,
        input_keep_prob=1.0,
        output_keep_prob=1.0
    )
    estimator = Estimator(
        model_fn=model_fn,
        model_dir=MODEL_DIR,
        config=tf.contrib.learn.RunConfig())

    input_fn_train = cn_inputs.create_input_fn(
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        input_files=[TRAIN_FILE],
        batch_size=hparams.batch_size,
        num_epochs=FLAGS.num_epochs)
    input_fn_eval = cn_inputs.create_input_fn(
        mode=tf.contrib.learn.ModeKeys.EVAL,
        input_files=[VALIDATION_FILE],
        batch_size=hparams.eval_batch_size,
        num_epochs=1)
    # tf.contrib.learn.RunConfig()

    eval_metrics = cn_metrics.create_evaluation_metrics()

    eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=input_fn_eval,
        every_n_steps=FLAGS.eval_every,
        metrics=eval_metrics,
        early_stopping_metric="recall_at_1",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=3000
    )  # 喂数据

    estimator.fit(input_fn=input_fn_train, steps=None, monitors=[eval_monitor])


if __name__ == "__main__":
    tf.app.run()
