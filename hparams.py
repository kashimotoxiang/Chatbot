import tensorflow as tf
import os
from collections import namedtuple
'''
开关
'''
tf.flags.DEFINE_boolean("customized_word_vector", True,
                        "choose random or customized word vectors")

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
文件夹
'''
# runs文件夹
tf.flags.DEFINE_string("RNN_CNN_MaxPooling_model_dir", 'runs/RNN_CNN_MaxPooling',
                       "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_string("RNN_MaxPooling_model_dir", 'runs/RNN_MaxPooling',
                       "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_string("RNN_model_dir", 'runs/RNN',
                       "Directory to store model checkpoints (defaults to ./runs)")

# 输入输出文件夹
tf.flags.DEFINE_string("input_dir", os.path.abspath("data/source"),
                       "Directory containing input data files")

tf.flags.DEFINE_string(
    "output_dir", os.path.abspath("data/new_data"),
    "Output directory")

# 'word2vec/word2vec.npy'
tf.flags.DEFINE_string("word2vec_path", os.path.abspath(os.path.join(tf.flags.FLAGS.input_dir, 'word2vec.npy')),
                       "Path to word2vec.npy file")

tf.flags.DEFINE_string("vocab_path", None, "Path to vocabulary.txt file")
tf.flags.DEFINE_string("vocab_processor_file", os.path.abspath(os.path.join(
    tf.flags.FLAGS.input_dir, 'vocab_processor.bin')), "Saved vocabulary processor file")

tf.flags.DEFINE_string("initial_embeddings_path", os.path.join(
    tf.flags.FLAGS.output_dir, "initial_embeddings.npy"),
    "Path to initial_embeddings.npy file")

FLAGS = tf.flags.FLAGS

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
