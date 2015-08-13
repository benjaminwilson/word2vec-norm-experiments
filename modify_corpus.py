"""
Modifies an input text for the experiments according to the parameters defined in parameters.py
Assumes the filenames from filenames.sh
Requires sufficient diskspace to write out the modified text at intermediate steps.
"""
from __future__ import print_function
import os
from parameters import *
from functions import *

directory = os.path.dirname(os.path.realpath(__file__))
filenames = dict()
execfile(os.path.join(directory, 'filenames.sh'), filenames)

def read_words(filename):
    words = []
    with file(filename) as f:
        for line in f:
            word = line.split(',')[0]
            words.append(word)

wf_experiment_words = read_words(filenames['word_frequency_experiment_words']) #FIXME change names
cn_experiment_words = read_words(filenames['coocc_noise_experiment_words'])

counts = dict()
with file(filenames['word_counts']) as f:
    for line in f:
        word, count = line.strip().split(',')
        counts[word] = int(count)
total_words = sum(counts.values())

# intersperse the meaningless token throughout the corpus
intermediate_file = 'delete.me'
with open(filenames['corpus_unmodified']) as f_in, open(intermediate_file, 'w') as f_out:
    intersperse_words({meaningless_token: meaningless_token_frequency}, f_in, f_out)
word_frequency_experiment_words.append(meaningless_token)

# perform the replacement procedures for the word frequency and the noise cooccurrence experiments
word_samplers = {}
for word in wf_experiment_words:
    word_samplers[word] = truncated_geometric_sampling(word, word_freq_experiment_ratio, word_freq_experiment_power_max)
for word in cn_experiment_words:
    word_samplers[word] = truncated_geometric_sampling(word, coocc_noise_experiment_ratio, coocc_noise_experiment_power_max)

tmp_file = 'delete.me.2'
with open(intermediate_file) as f_in, open(tmp_file, 'w') as f_out:
    replace_words(word_samplers, f_in, f_out)
os.remove(intermediate_file)
intermediate_file = tmp_file

# add noise to the cooccurrence distributions
token_freq_dict = dict()
for word in cn_experiment_words:
    target_freq = counts[word] * 1. / total_words
    for i in range(1, coocc_noise_experiment_power_max + 1):
        current_freq = target_freq * truncated_geometric_proba(coocc_noise_experiment_ratio, i, coocc_noise_experiment_power_max)
        token_freq_dict[build_experiment_token(word, i)] = target_freq - current_freq

with open(intermediate_file) as f_in, open(filenames['corpus_modified'], 'w') as f_out:
    intersperse_words(token_freq_dict, f_in, f_out)
os.remove(intermediate_file)
