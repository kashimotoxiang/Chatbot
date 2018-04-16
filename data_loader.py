import tensorflow as tf
import hparams

TEXT_FEATURE_SIZE = 160
FLAGS = tf.flags.FLAGS


class DataLoader():
    _sess = None
    _trarning_path = None
    _validation_path = None
    _test_path = None

    def __init__(self, _sess, _trarning_path, _validation_path, _test_path):
        self._sess = _sess
        self._trarning_path = _trarning_path
        self._validation_path = _validation_path
        self._test_path = _test_path

    def load_train_data(self):
        feature_map, target = self.create_input_fn(tf.contrib.learn.ModeKeys.TRAIN,
                                                   self._trarning_path,
                                                   FLAGS.batch_size,
                                                   FLAGS.num_epochs)
        return feature_map, target

    def load_eval_data(self):
        feature_map, target = self.create_input_fn(tf.contrib.learn.ModeKeys.EVAL,
                                                   self._validation_path,
                                                   FLAGS.batch_size,
                                                   FLAGS.num_epochs)
        return feature_map, target

    def create_input_fn(self, mode, input_files, batch_size, num_epochs):
        features = tf.contrib.layers.create_feature_spec_for_parsing(
            self.get_feature_columns(mode))
        # 从输入文件创建batch
        feature_map = tf.contrib.learn.io.read_batch_features(
            file_pattern=input_files,
            batch_size=batch_size,
            features=features,
            reader=tf.TFRecordReader,
            randomize_input=True,
            num_epochs=num_epochs,
            queue_capacity=200000 + batch_size * 10,  # 洗牌 shuffle
            name="read_batch_features_{}".format(mode))

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            target = feature_map.pop("label")
        else:
            # In evaluation we have 10 classes (utterances).
            # The first one (index 0) is always the correct one
            target = tf.zeros([batch_size, 1], dtype=tf.int64)

        iterator = feature_map['context']
        context_batch = self._sess.run(iterator)
        return context_batch


    def get_feature_columns(self, mode):
        feature_columns = []
        # 输进去一个问题，问题长度和答案长度
        feature_columns.append(tf.contrib.layers.real_valued_column(
            column_name="context", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))  # 每一个问题和答案最大长度
        feature_columns.append(tf.contrib.layers.real_valued_column(
            column_name="context_len", dimension=1, dtype=tf.int64))
        feature_columns.append(tf.contrib.layers.real_valued_column(
            column_name="utterance", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
        feature_columns.append(tf.contrib.layers.real_valued_column(
            column_name="utterance_len", dimension=1, dtype=tf.int64))

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            # During training we have a label feature
            feature_columns.append(tf.contrib.layers.real_valued_column(
                column_name="label", dimension=1, dtype=tf.int64))

        if mode == tf.contrib.learn.ModeKeys.EVAL:
            # During evaluation we have distractors
            for i in range(9):
                feature_columns.append(tf.contrib.layers.real_valued_column(
                    column_name="distractor_{}".format(i), dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))  # 输入9个错误答案
                feature_columns.append(tf.contrib.layers.real_valued_column(
                    column_name="distractor_{}_len".format(i), dimension=1, dtype=tf.int64))

        return set(feature_columns)
