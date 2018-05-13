import csv
import os
import numpy as np
import tensorflow as tf
import cn_model
import cn_hparams
from models.model import encoder_model
from models import model
from data import file_process as fp
from make_data import Dataset
from make_data import Tokenized
import functools
import lyx.common as lyx

use_word2vec = True


# model=model.RNN_CNN_MaxPooling
model = functools.partial(model.RNN_CNN_MaxPooling, filtersizes=[2, 3, 4, 5],
                          num_filters=60)

if isinstance(model, functools.partial):
    model.name = model.func.__name__
else:
    model.name = model.__name__


RNNInit = tf.nn.rnn_cell.LSTMCell
is_bidirection = True

cn_hparams.model_dir_generator(
    use_word2vec, model.name+'60', RNNInit.__name__, is_bidirection)
vocab=lyx.load_pkl("vocab")

tf.logging.set_verbosity(tf.logging.ERROR)
FLAGS = tf.flags.FLAGS
MODEL_DIR = FLAGS.RUNS

# restore vocab processor
filepath = os.path.join(FLAGS.make_data_output_dir, 'vocab_processor.bin')
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(filepath)

# Load your own data here
filepath = os.path.join(FLAGS.make_data_output_dir, "tokenized.pkl")
tokenized = fp.load_obj(filepath)

CONTEXT_INDEX = 13
INPUT_CONTEXT = tokenized.tokenized_context_list[CONTEXT_INDEX]


def get_utterance_features(tokenized, utterence_range):
    utterance_matrix = tokenized.tokenized_utterence_list[utterence_range[0]:utterence_range[1]]
    utterance_len = tokenized.tokenized_utterence_len_list[utterence_range[0]:utterence_range[1]]
    features = {
        "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
        "utterance_len": tf.constant(utterance_len, shape=[len(utterance_len), 1], dtype=tf.int64),
    }
    return features


def get_features(tokenized, input, utterence_range):

    context = next(vp.transform([input])).tolist()
    context_len =len(next(vocab._tokenizer([input])))
    utterance_matrix = tokenized.tokenized_utterence_list[utterence_range[0]:utterence_range[1]]
    utterance_len = tokenized.tokenized_utterence_len_list[utterence_range[0]:utterence_range[1]]

    context_matrix = np.array(
        [context] * (utterence_range[1]-utterence_range[0]))
    context_len = np.array(
        [context_len] * (utterence_range[1]-utterence_range[0]))
    features = {
        "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
        "context_len": tf.constant(context_len, shape=[len(context_len), 1], dtype=tf.int64),
        "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
        "utterance_len": tf.constant(utterance_len, shape=[len(utterance_len), 1], dtype=tf.int64),
    }
    # features["context"]=tf.convert_to_tensor(context_matrix, dtype=tf.int64)
    # features["context_len"]=tf.constant(context_len, shape=[len(context_len), 1], dtype=tf.int64)
    return features, None


if __name__ == "__main__":

    # get raw data
    dataset=lyx.load_pkl('dataset')
    raw_data=dataset.raw_data

    # restore model & parameters
    hparams = cn_hparams.create_hparams()
    model_fn = cn_model.create_model_fn(
        hparams,
        model_impl=encoder_model,
        model_fun=model,
        RNNInit=RNNInit,
        is_bidirection=is_bidirection,
        input_keep_prob=1.0,
        output_keep_prob=1.0
    )
    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn, model_dir=MODEL_DIR)

    while True:
        input_question = input('欢迎咨询医疗问题，请描述您的问题: ')


        # if input the index of question
        if input_question.isdigit():
            input_question_index = int(input_question)
            input_question=raw_data[input_question_index]['question']

        print('您的问题：%s'%input_question)

        probs = np.array(list(estimator.predict(input_fn=lambda: get_features(tokenized, input_question, [0, 100]))))
        sort_probs_idx = np.argsort(-probs, 0).flatten()
        print(sort_probs_idx[:100])
        top_index = sort_probs_idx[:10]
        max_probs = np.max(probs)
        print('结合您的描述：')
        for index in top_index:
            print(str(index)+"\t"+raw_data[index]["question"]+"\t"+raw_data[index]["answer"])

    # max_index = np.argmax(probs)
    # max_probs = np.max(probs)

    # print("answer_id: {max_index} probs:{max_probs}".format(
    #     max_index=max_index, max_probs=max_probs))
    # print('\nmatrices of probs:')
    # print(max_probs)
    #
    # print('question: {}'.format(raw_data[CONTEXT_INDEX]["question"]))
    # print('right answer: {}'.format(raw_data[CONTEXT_INDEX]["answer"]))
    # print('The most likely answer: {}'.format(raw_data[max_index]["answer"]))
    # print('rank of right answer is: {}'.format(right_answer_rank))
