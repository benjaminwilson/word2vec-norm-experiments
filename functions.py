from __future__ import print_function
import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
import matplotlib.pyplot as plt
from collections import defaultdict
from parameters import random_seed, experiment_word_occurrence_min

def count_words(f):
    """
    Return a dictionary mapping each word to its occurrence count in the file specified.
    Words are separated by ' '.
    """
    counts = defaultdict(lambda: 0)
    for line in f:
        for word in line.strip().split(' '):
                counts[word] += 1
    return counts

def count_document_frequency(f, vocab):
    """
    Return a dictionary mapping each word to the number of documents it occurs
    in, in the file 'f' specified.  Documents are separated from one another by
    linefeeds.
    Words are separated by ' '.
    """
    counts = {word: 0 for word in vocab}
    for line in f:
        for word in set(line.strip().split(' ')):
                counts[word] += 1
    return counts

def read_words(filename):
    """
    given a CSV of word counts, returns the list of words.
    """
    words = []
    with file(filename) as f:
        for line in f:
            word = line.split(',')[0]
            words.append(word)
    return words

def read_word_counts(f):
    """
    given a CSV of word counts, returns a word count dictionary.
    """
    counts = dict()
    for line in f:
        word, count = line.strip().split(',')
        counts[word] = int(count)
    return counts

def build_experiment_token(word, sample):
    """
    e.g. ('cat', 3) -> CAT_3
    """
    return '%s_%i' % (word.upper(), sample)

def truncated_geometric_proba(ratio, i, n):
    """
    return the probability of i being sampled from [1 .. n] from the truncated geometric distribution with the given ratio,
    i.e. the unique distn such that the probabilities decrease by ratio each time, and all are non-zero.
    """
    return (ratio ** (i - 1)) * (1 - ratio) / (1 - ratio ** n)

def truncated_geometric_sampling(word, ratio, max_value):
    """
    returns a function that samples from the truncated geometric distribution, truncated
    at max_value, returning WORD_1, .. WORD_<max_value>
    """
    outcomes = [build_experiment_token(word, value) for value in range(1, max_value + 1)]
    probs = np.array([truncated_geometric_proba(ratio, i, max_value) for i in range(1, max_value + 1)])
    return lambda: np.random.choice(outcomes, p=probs)

def intersperse_words(interspersal_rates, f_in, f_out):
    """
    Insperse words uniformly at random throughout the text in file-like object 'f_in',
    writing the result to file-like object 'f_out'.  'interspersal_rates' is a dict
    mapping words to the rate at which they should be interspersed, e.g.
        interspersal_rates = {'CAT_3': 0.004, 'MEANINGLESS': 0.0001}

    Our use of this function in the experiments assumes that total number of
    words remains essentially unchanged.
    """
    insertion_proba = sum(interspersal_rates.values())
    insertion_words = interspersal_rates.keys()
    relative_probas = np.array([interspersal_rates[word] for word in insertion_words]) / insertion_proba
    def sample_insertion_word():
        return np.random.choice(insertion_words, p=relative_probas)
        
    def must_insert():
        return random.random() < insertion_proba
    
    for line in f_in:
        words_out = []
        for word in line.strip().split(' '):
            if must_insert():
                words_out.append(sample_insertion_word())
            words_out.append(word)
        print(' '.join(words_out), file=f_out)

def replace_words(word_sampler_dict, f_in, f_out):
    """
    Performs a replacement procedure on the text in read from file-like object 'f_in', writing the
    results to the file-like object 'f_out'.  'word_sampler_dict' is a dict mapping words to
    be replaced to functions (without arguments) that return their replacement.
    e.g. word_sampler_dict = {'cat': truncated_geometric_proba('cat', 0.5, 20)} 
    """
    for line in f_in:
        words_out = []
        for word in line.strip().split(' '):
            if word in word_sampler_dict:
                sampler = word_sampler_dict[word]
                word = sampler()
            if word is not None:
                words_out.append(word)
        print(' '.join(words_out), file=f_out)

def load_word2vec_binary(fname):
    """
    Loads a word2vec word vectors binary file, returns DataFrame.

    Method from:
    http://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py
    """
    vocab = []
    vectors = None

    with open(fname) as fin:
        header = fin.readline()
        vocab_size, vector_size = map(int, header.split())

        vectors = np.empty((vocab_size, vector_size), dtype=np.float)
        binary_len = np.dtype(np.float32).itemsize * vector_size
        for line_no in xrange(vocab_size):
            word = ''
            while True:
                ch = fin.read(1)
                if ch == ' ':
                    break
                word += ch
            vocab.append(word.strip())

            vector = np.fromstring(fin.read(binary_len), np.float32)
            vectors[line_no] = vector
    return pd.DataFrame(vectors, index=vocab)

def cosine_similarity(vecs):
    """
    return the cosine similarity of each row with all the others.
    'vecs' is a dataframe
    """
    vecs_normed = vecs.as_matrix() / np.sqrt((vecs ** 2).sum(axis=1))[:,np.newaxis]
    return vecs_normed.dot(vecs_normed.transpose())    

def cosine_similarity_heatmap(test_vecs, ticks, **kwargs):
    mat = cosine_similarity(test_vecs)
    plt.figure(**kwargs)
    plt.title('Cosine similarity of word vectors')
    _ = plt.gca().set_ylim(0, len(test_vecs.index))
    _ = plt.gca().set_xlim(0, len(test_vecs.index))
    plt.pcolor(mat, vmin=-1, vmax=1)
    plt.colorbar()
    _ = plt.yticks(np.arange(0.5, len(test_vecs.index), 1), ticks, fontsize=11)
    _ = plt.xticks(np.arange(0.5, len(test_vecs.index), 1), ticks, rotation=90, fontsize=11)
    plt.tight_layout()

def by_distance_from(table, v, **params):
    """
    Return a Series, listing the distance of each row
    from the vector given.
    To specify different metrics, see docstring of scipy.spatial.distance.cdist
    (default if Euclidean).
    """
    v = np.array(v).reshape(1, -1)
    dist = pd.Series(
        cdist(v, table, **params)[0], index=table.index, copy=True)
    dist.sort()
    return dist

def row_normalise_matrix(matrix):
    """
    All matrices are represented by 2 dimension np.array instances (NOT np.matrix)
    return the matrix, along with the norms.
    """
    matrix = matrix * 1.
    norms = np.sqrt((matrix ** 2).sum(axis=1))
    return matrix / norms[:, np.newaxis], norms

def row_normalise_dataframe(df):
    """
    As per normalise_matrix, but accepts and returns a DataFrame and Series in
    place of arrays.  The DataFrame and Series share the index of df.
    """
    normed_mat, norms = row_normalise_matrix(df.as_matrix())
    normed_df = pd.DataFrame(normed_mat, index=df.index, columns=df.columns)
    norms_series = pd.Series(norms, index=df.index)
    return normed_df, norms_series
