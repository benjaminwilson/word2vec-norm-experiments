from __future__ import print_function
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from parameters import coocc_noise_experiment_freq_reduction

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

def read_words(f):
    """
    given a CSV of word counts, returns the list of words.
    """
    words = []
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
    return the probability of i being sampled from [1 .. n] from the truncated
    geometric distribution with the given the ratio, i.e. the unique distn such
    that the probabilities decrease by ratio each time, and all are non-zero.
    """
    return (ratio ** (i - 1)) * (1 - ratio) / (1 - ratio ** n)

def distribution_to_sampling_function(word, dist_fn, max_value):
    """
    returns a function that samples from the given distribution 'dist_fn' on
    [1 .. max_value], returning from [WORD_1, .. WORD_<max_value>]
    """
    outcomes = range(1, max_value + 1)
    tokens = [build_experiment_token(word, value) for value in outcomes]
    probs = np.array([dist_fn(i) for i in outcomes])
    return lambda: np.random.choice(tokens, p=probs)

def evenly_spaced_proba(i, M): 
    """
    A probability distribution on [1 .. M] with the property that the
    probability densities are evenly spaced, i.e. p(i) - p(i+1) = c for all i.
    The sequence is decreasing, so c > 0.
    p(M) = 0
    """
    return 2. * (M - i) / (M * (M-1))

def noise_proportion(i, M):
    """
    Assuming that the evenly_spaced_proba(i,M) distribution was used and that
    the total number of occurrences (original + noise) is given by
        #original * coocc_noise_experiment_freq_reduction
    irrespective of i, return the expected proportion of
    noise occurrences.
    """
    return 1 - evenly_spaced_proba(i, M) / coocc_noise_experiment_freq_reduction

def intersperse_words(interspersal_rates, f_in, f_out):
    """
    Insperse words uniformly at random throughout the text in file-like object
    'f_in', writing the result to file-like object 'f_out'.
    'interspersal_rates' is a dict mapping words to the rate at which they
    should be interspersed, e.g.
        interspersal_rates = {'CAT_3': 0.004, 'MEANINGLESS': 0.0001}

    Our use of this function in the experiments assumes that total number of
    words remains essentially unchanged by the interspersal.
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
    Performs a replacement procedure on the text read in from file-like object
    'f_in', writing the results to the file-like object 'f_out'.
    'word_sampler_dict' is a dict mapping words to be replaced to functions
    (without arguments) that return their replacement.  e.g.
    word_sampler_dict = {'cat': distribution_to_sampling_function('cat', dist_fn, 20)}
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
    return the cosine similarity of each row (vector) with all the others.
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
