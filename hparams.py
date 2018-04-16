import tensorflow as tf
import os
from collections import namedtuple

FLAGS = tf.flags.FLAGS


def define_abs_join_path(name, predix_dir, dir, comment=''):
    return tf.flags.DEFINE_string(name, os.path.abspath(os.path.join(predix_dir, dir)), comment)


'''
开关
'''
tf.flags.DEFINE_boolean("customized_word_vector", True,
                        "choose random or customized word vectors")


# 参数选择：
# 'RNN_CNN_MaxPooling'
# 'RNN_MaxPooling'
# 'RNN'
tf.flags.DEFINE_string("model_name", 'RNN_CNN_MaxPooling',
                        "choose random or customized word vectors")

# runs文件夹
if FLAGS.model_name == 'RNN_CNN_MaxPooling':
    define_abs_join_path("RUNS", '', 'runs/RNN_CNN_MaxPooling')
elif FLAGS.model_name == 'RNN_MaxPooling':
    define_abs_join_path("RUNS", '', 'runs/RNN_MaxPooling')
elif FLAGS.model_name == 'RNN':
    define_abs_join_path("RUNS", '', 'runs/RNN')


'''
超参数
'''
tf.flags.DEFINE_integer(
    "vocab_size",
    3389,
    "The size of the vocabulary. Only change this if you changed the preprocessing")
# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 300,
                        "Dimensionality of the embeddings")
tf.flags.DEFINE_integer("rnn_dim", 256, "Dimensionality of the RNN cell")
tf.flags.DEFINE_integer("max_context_len", 160,
                        "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 80,
                        "Truncate utterance to this length")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 8, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam",
                       "Optimizer Name (Adam, Adagrad, etc)")

tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None,
                        "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 1,
                        "Evaluate after this many train steps")

tf.flags.DEFINE_integer("min_word_frequency", 5,
                        "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")

tf.flags.DEFINE_integer(
    "distraction_num", 9,
    "Output directory for TFRecord files (default = './data')")


'''
生成数据集文件夹
'''
# 输入输出文件夹
define_abs_join_path("make_data_input_dir", '', "data/source")
define_abs_join_path("make_data_output_dir", '', "data/new_data")


# 词向量等数据文件夹
# 'word2vec/word2vec.npy'
define_abs_join_path("word2vec_path",
                     FLAGS.make_data_input_dir, "word2vec.npy")
define_abs_join_path("vocab_path",
                     FLAGS.make_data_output_dir, 'vocabulary.txt')
define_abs_join_path("vocab_processor_file",
                     FLAGS.make_data_output_dir, "vocab_processor.bin")
define_abs_join_path("initial_embeddings_path",
                     FLAGS.make_data_output_dir, "initial_embeddings.npy")

'''
训练文件夹
'''
define_abs_join_path("input_dir", '', FLAGS.make_data_output_dir)

# data set 文件夹
define_abs_join_path("TRAIN_FILE",
                     FLAGS.input_dir, "train.tfrecords")
define_abs_join_path("VALIDATION_FILE",
                     FLAGS.input_dir, "validation.tfrecords")
define_abs_join_path("TEST_FILE",
                     FLAGS.input_dir, "test.tfrecords")


# checkpoint 文件夹
define_abs_join_path("CHECKPOINT_DIR",
                     FLAGS.RUNS, "checkpoints")


HParams = namedtuple(
    "HParams",
    [
        "batch_size",
        "embedding_dim",
        "eval_batch_size",
        "learning_rate",
        "max_context_len",
        "max_utterance_len",
        "optimizer",
        "rnn_dim",
        "vocab_size",
        "vocab_path",
        "word2vec_path",

    ])


def create_hparams():
    return HParams(
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        vocab_size=FLAGS.vocab_size,
        optimizer=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate,
        embedding_dim=FLAGS.embedding_dim,
        max_context_len=FLAGS.max_context_len,
        max_utterance_len=FLAGS.max_utterance_len,
        vocab_path=FLAGS.vocab_path,
        word2vec_path=FLAGS.word2vec_path,
        rnn_dim=FLAGS.rnn_dim)