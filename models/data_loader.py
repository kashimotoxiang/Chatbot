import array
import numpy as np
import tensorflow as tf
from collections import defaultdict
import os


def load_vocab(filename):
    vocab = None
    with open(filename, encoding="UTF-8") as f:
        vocab = f.read().splitlines()
    dct = defaultdict(int)
    for idx, word in enumerate(vocab):
        dct[word] = idx
    return [vocab, dct]


def load_glove_vectors(filename, vocab):
    """
    Load glove vectors from a .txt file.
    Optionally limit the vocabulary to save memory. `vocab` should be a set.
    """

    dct = {}
    vectors = array.array('d')
    current_idx = 0

    if os.path.splitext(filename)[1] == '.npy':
        word2vec = np.load(filename)
        for tokens in word2vec:
            # tokens = line.split(" ")
            word = tokens[0]
            entries = tokens[1:]
            if not vocab or word in vocab:
                dct[word] = current_idx
                vectors.extend(float(x) for x in entries)
                current_idx += 1
    elif os.path.splitext(filename)[1] == '.txt':
        with open(filename, "r", encoding="utf-8") as f:
            for _, line in enumerate(f):
                tokens = line.split(" ")
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
    # np.save('./data/vec', vec)
    # np.save('./data/dct', dct)
    # assert False
    return [vec, dct]


def build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors, embedding_dim):
    initial_embeddings = np.random.uniform(
        -0.25, 0.25, (len(vocab_dict), embedding_dim)).astype("float32")  # 二维

    for word, glove_word_idx in glove_dict.items():
        word_idx = vocab_dict.get(word)
        initial_embeddings[word_idx, :] = glove_vectors[glove_word_idx]

    return initial_embeddings