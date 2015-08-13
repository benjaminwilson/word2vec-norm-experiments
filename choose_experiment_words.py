"""
Modifies an input text for the experiments according to the parameters defined in parameters.py
Assumes the filenames from filenames.sh
Writes out files listing the words chosen.
"""
from __future__ import print_function
import os
from parameters import *
from functions import *

directory = os.path.dirname(os.path.realpath(__file__))

filenames = dict()
execfile(os.path.join(directory, 'filenames.sh'), filenames)

counts = dict()
with file(filenames['word_counts']) as f:
    for line in f:
        word, count = line.strip().split(',')
        counts[word] = int(count)

frequent_words = [word for word in counts if counts[word] > experiment_word_occurrence_min]
random.seed(random_seed)
words_experiment_1, words_experiment_2 = [random.sample(frequent_words, number_of_experiment_words) for _ in range(2)]
words_experiment_1.append('the')

with file(filenames['word_freq_experiment_words'], 'w') as f:
    for word in words_experiment_1:
        print('%s,%i' % (word, counts[word]), file=f)

with file(filenames['coocc_noise_experiment_words'], 'w') as f:
    for word in words_experiment_2:
        print('%s,%i' % (word, counts[word]), file=f)
