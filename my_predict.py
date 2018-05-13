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


def get_raw_data(filename):
    with open(filename, "r", encoding="UTF-8") as csvFile:
        reader = csv.reader(csvFile)
        raw_data = [{
            'question': row[1], 'answer': row[2]} for row in reader if len(row[1]) != 0]
    return raw_data


tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_string("model_dir", "./runs/biLSTM",
#                        "Directory to load model checkpoints from")
MODEL_DIR = FLAGS.RUNS

FLAGS = tf.flags.FLAGS


def tokenizer_fn(iterator):
    return (x.split(" ") for x in iterator)


# Load vocabulary
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


def get_features(tokenized, context, utterence_range):
    if isinstance(context, str):
        context = next(vp.transform([context])).tolist()
    utterance_matrix = tokenized.tokenized_utterence_list[utterence_range[0]:utterence_range[1]]
    utterance_len = tokenized.tokenized_utterence_len_list[utterence_range[0]:utterence_range[1]]
    context_len = len(context)
    context_matrix = np.array([context] * (utterence_range[1] - utterence_range[0]))
    context_len = np.array([context_len] * (utterence_range[1] - utterence_range[0]))
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

    raw_data = get_raw_data("data/corpus/aftercleaning_excludesubtitle.csv")

    hparams = cn_hparams.create_hparams()
    model_fn = cn_model.create_model_fn(
        hparams,
        model_impl=encoder_model,
        model_fun=model.RNN_CNN_MaxPooling,
        RNNInit=tf.nn.rnn_cell.LSTMCell,
        is_bidirection=True,
        input_keep_prob=1.0,
        output_keep_prob=1.0
    )
    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn, model_dir=MODEL_DIR)

    input_question = None

    while True:
        input_question = input('欢迎咨询医疗问题，请描述您的问题: ')

        probs = np.array(list(estimator.predict(
            input_fn=lambda: get_features(tokenized, input_question, [0, 100]))))
        sort_probs_idx = np.argsort(-probs, 0).flatten()
        print(sort_probs_idx)
        max_index = sort_probs_idx[0]
        max_probs = np.max(probs)

        if input_question.isdigit():
            print('测试系统: ')
            # print(sort_probs_idx)
            input_question = int(input_question)
            right_answer_rank = np.where(sort_probs_idx == input_question)[0][0]
            print('question: {}'.format(raw_data[input_question]["question"]))
            print('right answer: {}'.format(raw_data[input_question]["answer"]))
            print('The most likely answer: {}'.format(raw_data[max_index]["answer"]))
            print('rank of right answer is: {}'.format(right_answer_rank))
            print('right answer probs={},rank 1 answer probs={}'.format(
                probs[input_question],
                max_probs))
        else:
            print('结合您的描述：')
            print(raw_data[max_index]["answer"])

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
