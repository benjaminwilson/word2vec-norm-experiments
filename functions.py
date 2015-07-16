from __future__ import print_function
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from parameters import random_seed, experiment_word_occurrence_min

def count_words(f):
    """
    Return a dictionary mapping each word to its occurrence count in the file specified.
    Words are separated by ' '.
    """
    counts = defaultdict(lambda: 0)
    for line in f:
        for word in line.split(' '):
                counts[word] += 1
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
    Insperse words uniformly at random throughout the text in 'in_filename',
    writing the result to 'out_filename'.  'interspersal_rates' is a dict
    mapping words to the rate at which they should be interspersed, e.g.
        interspersal_rates = {'CAT_3': 0.004, 'MEANINGLESS': 0.0001}

    Our use of this function in the experiments assumes that total number of
    words remains essentially unchanged.
    """ #FIXME update doc
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
    Performs a replacement procedure on the text in 'in_filename', writing the
    results to 'out_filename'.  'word_sampler_dict' is a dict mapping words to
    be replaced to functions (without arguments) that return their replacement.
    e.g. word_sampler_dict = {'cat': truncated_geometric_proba('cat', 0.5, 20)} 
    """ #FIXME update doc
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


