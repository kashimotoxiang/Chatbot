# =================================
# using for initialize data sets
# =================================
import jieba
import numpy as np
import progressbar
import csv
from math import floor
from data import file_process as fp
import tensorflow as tf
import os
from hanziconv import HanziConv
import functools
import cn_hparams
import random
from collections import defaultdict
import array
import copy
import lyx.common as lyx

FLAGS = tf.flags.FLAGS


class Dataset():
    # private data
    _stopwordset = ''

    _counter = 0
    _bar = progressbar.ProgressBar(maxval=100)
    raw_data = []
    _data_len = 0

    _split_scope_sum = 0
    _split_scope_1 = 0
    _split_scope_2 = 0

    # public data
    train_data = []
    valid_data = []
    test_data = []

    tokenized_context_list = []
    tokenized_context_len_list = []
    tokenized_utterence_list = []
    tokenized_utterence_len_list = []
    id2vec_lookup_list = []
    word2id_lookup_list = {}
    # path
    _userdict = ''

    def __init__(self, *, filename, user_dict, stopword_dict, prop):
        self.tokenized_context_list = []
        self.tokenized_context_len_list = []
        self.tokenized_utterence_list = []
        self.tokenized_utterence_len_list = []
        # 初始化停用词表
        if stopword_dict is not []:
            self.set_stopword(stopword_dict)

        # 初始化自定义词表
        if user_dict is not []:
            self._userdict = user_dict

        jieba.load_userdict(self._userdict)

        # 导入数据集
        # 读取csv, 创建dataframe
        # print('导入数据集')
        with open(filename, "r", encoding="UTF-8") as csvFile:
            reader = csv.reader(csvFile)
            self.raw_data = [{
                'question': row[1], 'answer': row[2]} for row in reader if len(row[1]) != 0]
            csvFile.close()
        self._data_len = len(self.raw_data)

        # 切分数据集
        self.split_data_set(prop, self.raw_data, self._data_len)

    def split_data_set(self, prop, data, length):
        """
        分配数据集
        """
        self._split_scope_sum = sum(prop)
        self._split_scope_1 = prop[0] / self._split_scope_sum
        self._split_scope_2 = prop[1] / self._split_scope_sum

        random.shuffle(data)
        train_data_begin = 0
        train_data_end = train_data_begin + floor(self._split_scope_1 * length)
        valid_data_begin = train_data_end
        valid_data_end = valid_data_begin + floor(self._split_scope_2 * length)
        test_data_begin = valid_data_end
        test_data_end = length

        self.train_data = copy.deepcopy(data[train_data_begin:train_data_end])
        self.test_data = copy.deepcopy(data[valid_data_begin:valid_data_end])
        self.valid_data = copy.deepcopy(data[test_data_begin:test_data_end])
        self.off_train_data = copy.deepcopy(
            data[valid_data_begin:test_data_end])

        def map_neg_utterance(i, _data=self.train_data):
            data_len = len(_data)
            try:
                utterance_index = np.random.random_integers(
                    1, length) + i  # 避免撞上自己
                utterance_index = utterance_index % data_len
                utterance = _data[utterance_index]['answer']
                return utterance
            except Exception as e:
                print(e)

                # add neg sample into training data

        def map_train_data(data):
            data['label'] = 1
            return data

        pos_train_data = list(map(map_train_data, self.train_data))
        neg_utterance = list(
            map(map_neg_utterance, range(len(self.train_data))))

        neg_train_data = [{
            'question': self.train_data[i]['question'], 'answer': neg_utterance[i], 'label': 0} for i in
            range(len(self.train_data))]
        self.train_data = pos_train_data + neg_train_data
        random.shuffle(self.train_data)

        # Distractor sequences

        def map_test_data(_data):
            for i in range(FLAGS.distraction_num):
                try:
                    dis_key = "distractor_{}".format(i)
                    # Distractor Text Feature
                    dis_text = map_neg_utterance(i, _data=self.off_train_data)
                    _data[dis_key] = dis_text
                except Exception as e:
                    print(e)

            return _data

        self.test_data = list(map(map_test_data, self.test_data))
        self.valid_data = list(map(map_test_data, self.valid_data))

    def set_stopword(self, files):
        """
        load stop words
        """
        try:
            stopwords = ''
            for item in files:
                stopwords = ''.join(stopwords + '\n' +
                                    fp.readfile(item))
            stopwordslist = stopwords.split('\n')
            not_stopword = set(['', '\n'])
            self._stopwordset = set(stopwordslist)
            self._stopwordset = self._stopwordset - not_stopword
        except Exception as e:
            print(e)

    def movestopwords(self, sentence):
        '''
        remove stop words
        '''
        try:
            def is_stopwords(word):
                if (word != '\t' and '\n') and (word not in self._stopwordset):
                    return word and word.strip()

            res = list(filter(is_stopwords, sentence))
            return res
        except Exception as e:
            print(e)

    def chinese_tokenizer(self, documents):
        for document in documents:
            # 繁体转简体
            text = HanziConv.toSimplified(document)
            # 分词
            text = jieba.lcut(text)
            # 去除停用词
            if self._stopwordset:
                text = self.movestopwords(text)
            yield text

    def create_csv_iter(self, filename):
        """
        Returns an iterator over a CSV file. Skips the header.
        """
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            # Skip the header
            next(reader)
            for row in reader:
                yield row

    def create_vocab(self, input_iter, min_frequency):
        """
        Creates and returns a VocabularyProcessor object with the vocabulary
        for the input iterator.
        """
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
            FLAGS.max_sentence_len,
            min_frequency=min_frequency,
            tokenizer_fn=self.chinese_tokenizer,
            # vocabulary=self.word2id_lookup_list
        )
        vocab_processor.fit(input_iter)
        return vocab_processor

    def transform_sentence(self, sequence, vocab_processor):
        """
        Maps a single sentence into the integer vocabulary. Returns a python array.
        """
        return next(vocab_processor.transform([sequence])).tolist()

    def create_text_sequence_feature(self, fl, sentence, sentence_len, vocab):
        """
        Writes a sentence to FeatureList protocol buffer
        """
        sentence_transformed = self.transform_sentence(sentence, vocab)
        for word_id in sentence_transformed:
            fl.feature.add().int64_list.value.extend([word_id])
        return fl

    def create_example_train(self, row, vocab):
        """
        Creates a training example for the Ubuntu Dialog Corpus dataset.
        Returnsthe a tensorflow.Example Protocol Buffer object.
        """
        context, utterance, label = row
        context_transformed = self.transform_sentence(context, vocab)
        utterance_transformed = self.transform_sentence(utterance, vocab)
        context_len = len(next(vocab._tokenizer([context])))
        utterance_len = len(next(vocab._tokenizer([utterance])))
        label = int(float(label))

        # New Example
        example = tf.train.Example()
        example.features.feature["context"].int64_list.value.extend(
            context_transformed)
        example.features.feature["utterance"].int64_list.value.extend(
            utterance_transformed)
        example.features.feature["context_len"].int64_list.value.extend([
            context_len])
        example.features.feature["utterance_len"].int64_list.value.extend([
            utterance_len])
        example.features.feature["label"].int64_list.value.extend([label])
        return example

    def create_example_test(self, row, vocab):
        """
        Creates a test/validation example for the Ubuntu Dialog Corpus dataset.
        Returnsthe a tensorflow.Example Protocol Buffer object.
        """
        context, utterance = row[:2]
        distractors = row[2:]
        context_len = len(next(vocab._tokenizer([context])))
        utterance_len = len(next(vocab._tokenizer([utterance])))
        context_transformed = self.transform_sentence(context, vocab)
        utterance_transformed = self.transform_sentence(utterance, vocab)

        # New Example
        example = tf.train.Example()
        example.features.feature["context"].int64_list.value.extend(
            context_transformed)
        example.features.feature["utterance"].int64_list.value.extend(
            utterance_transformed)
        example.features.feature["context_len"].int64_list.value.extend([
            context_len])
        example.features.feature["utterance_len"].int64_list.value.extend([
            utterance_len])

        # Distractor sequences
        for i, distractor in enumerate(distractors):
            dis_key = "distractor_{}".format(i)
            dis_len_key = "distractor_{}_len".format(i)
            # Distractor Length Feature
            dis_len = len(next(vocab._tokenizer([distractor])))
            example.features.feature[dis_len_key].int64_list.value.extend([
                dis_len])
            # Distractor Text Feature
            dis_transformed = self.transform_sentence(distractor, vocab)
            example.features.feature[dis_key].int64_list.value.extend(
                dis_transformed)
        return example

    def create_tfrecords_file(self, data, output_filename, example_fn):
        """
        Creates a TFRecords file for the given input data and
        example transofmration function
        """
        with tf.python_io.TFRecordWriter(output_filename) as writer:
            print("Creating TFRecords file at {}...".format(output_filename))

            examples = map(lambda row: example_fn(
                row).SerializeToString(), data)

            print("Wrote to {}".format(output_filename))
            for item in examples:
                writer.write(item)

    def write_vocabulary(self, vocab_processor, outfile):
        with open(outfile, "w", encoding='utf-8') as vocabfile:
            list(map(lambda item: vocabfile.write('%s\n' % item),
                     vocab_processor.vocabulary_._reverse_mapping))
        print("Saved vocabulary to {}".format(outfile))


    def len_tokenizer(self, input):
        # 繁体转简体
        text = HanziConv.toSimplified(input)
        # 分词
        text = jieba.lcut(text)
        # 去除停用词
        if self._stopwordset:
            text = self.movestopwords(text)
        return len(text)





def create_embeddings(vocab_array, word2vec, embedding_dim):
    tf.logging.info("Greating Glove embeddings...")

    # load_vocab
    vocab_dict = defaultdict(int)
    for idx, word in enumerate(vocab_array):
        vocab_dict[word] = idx

    # load_glove_vectors
    vocab = set(vocab_array)
    dct = {}
    vectors = array.array('d')
    current_idx = 0

    for tokens in word2vec:
        word = tokens[0]
        entries = tokens[1:]
        if not vocab or word in vocab:
            dct[word] = current_idx
            vectors.extend(float(x) for x in entries)
            current_idx += 1
    word_dim = len(entries)
    num_vectors = len(dct)
    tf.logging.info("Found {} out of {} vectors in Glove".format(
        num_vectors, len(vocab)))
    vec = np.array(vectors).reshape(num_vectors, word_dim)

    # build_initial_embedding_matrix
    glove_vectors = vec
    glove_dict = dct
    initial_embeddings = np.random.uniform(
        -0.25, 0.25,
        (len(vocab_dict), embedding_dim)).astype("float32")  # 二维
    for word, glove_word_idx in glove_dict.items():
        word_idx = vocab_dict.get(word)
        initial_embeddings[word_idx, :] = glove_vectors[glove_word_idx]
    np.save(FLAGS.initial_embeddings_path, initial_embeddings)


class Tokenized():
    tokenized_context_list = None
    tokenized_context_len_list = None
    tokenized_utterence_list = None
    tokenized_utterence_len_list = None
    dataset = None
    vocab = None

    def __init__(self, dataset, vocab):
        self.dataset = dataset
        self.vocab = vocab
        raw_data=self.dataset.raw_data

        question_data = [x['question'] for x in raw_data]

        print('tokenized_context_list')
        tokenized_context_list = list(
            lyx.mp_map(self.tokenizer, question_data))
        print('tokenized_context_len_list')
        tokenized_context_len_list = list(
            lyx.mp_map(self.dataset.len_tokenizer, question_data))

        answer_data = [x['answer'] for x in raw_data]
        print('tokenized_utterence_list')
        tokenized_utterence_list = list(
            lyx.mp_map(self.tokenizer, answer_data))
        print('tokenized_utterence_len_list')
        tokenized_utterence_len_list = list(
            lyx.mp_map(self.dataset.len_tokenizer,  answer_data))

        self.tokenized_context_list = np.array(tokenized_context_list)
        self.tokenized_context_len_list = np.array(tokenized_context_len_list)
        self.tokenized_utterence_list = np.array(tokenized_utterence_list)
        self.tokenized_utterence_len_list = np.array(
            tokenized_utterence_len_list)

    def tokenizer(self, input):
        return self.dataset.transform_sentence(input, self.vocab)


if __name__ == "__main__":
    # dataset = Dataset(filename="data/corpus/aftercleaning_excludesubtitle.csv",
    #                   # aftercleaning_excludesubtitle
    #                   # corpus_small
    #                   user_dict="data/dict/自定义词典.txt",
    #                   stopword_dict=[
    #                       'data/dict/哈工大停用词表.txt',
    #                       'data/dict/自定义停用词.txt'],
    #                   prop=[0.95, 0.03, 0.02])
    #
    # print("Creating input...")
    # input_iter = (x['question'] + ' ' + x['answer'] for x in dataset.raw_data)
    # print("Creating vocabulary...")
    # vocab = dataset.create_vocab(
    #     input_iter, min_frequency=FLAGS.min_word_frequency)
    # print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))
    # lyx.save_pkl(vocab, "vocab")
    # lyx.save_pkl(dataset, "dataset")

    vocab=lyx.load_pkl("vocab")
    dataset=lyx.load_pkl("dataset")


    # # Create vocabulary.txt file
    # dataset.write_vocabulary(vocab, os.path.join(
    #     FLAGS.make_data_output_dir, 'vocabulary.txt'))
    #
    # # Save vocab processor
    # vocab.save(os.path.join(
    #     FLAGS.make_data_output_dir, 'vocab_processor.bin'))
    #
    # create_embeddings(
    #     vocab.vocabulary_._reverse_mapping,
    #     np.load(FLAGS.word2vec_path),
    #     FLAGS.embedding_dim)
    # # Create tfrecords
    # print("Creating validation tfrecords...")
    # input = ([x['question'], x['answer'], x['distractor_0'], x['distractor_1'], x['distractor_2'],
    #           x['distractor_3'], x['distractor_4'], x['distractor_5'],
    #           x['distractor_6'], x['distractor_7'], x['distractor_8'],
    #           ] for x in dataset.valid_data)
    # dataset.create_tfrecords_file(input,
    #                               output_filename=os.path.join(
    #                                   FLAGS.make_data_output_dir, "validation.tfrecords"),
    #                               example_fn=functools.partial(dataset.create_example_test, vocab=vocab))
    #
    # print("Creating test tfrecords...")
    # input = ([x['question'], x['answer'], x['distractor_0'], x['distractor_1'], x['distractor_2'],
    #           x['distractor_3'], x['distractor_4'], x['distractor_5'],
    #           x['distractor_6'], x['distractor_7'], x['distractor_8'],
    #           ] for x in dataset.test_data)
    # dataset.create_tfrecords_file(input,
    #                               output_filename=os.path.join(
    #                                   FLAGS.make_data_output_dir, "test.tfrecords"),
    #                               example_fn=functools.partial(dataset.create_example_test, vocab=vocab))  # 固定函数变量
    #
    # print("Creating training tfrecords...")
    # input = ([x['question'], x['answer'], x['label']]
    #          for x in dataset.train_data)
    # dataset.create_tfrecords_file(input,
    #                               output_filename=os.path.join(
    #                                   FLAGS.make_data_output_dir, "train.tfrecords"),
    #                               example_fn=functools.partial(dataset.create_example_train, vocab=vocab))
    #

    tokenized = Tokenized(dataset, vocab)

    lyx.save_pkl(tokenized, os.path.join(
        FLAGS.make_data_output_dir, "tokenized"))
