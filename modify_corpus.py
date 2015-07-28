"""
Modifies an input text for the experiments according to the parameters defined in parameters.py
Assumes the filenames from filenames.sh
Writes out files listing the words chosen.
Requires sufficient diskspace to write out the modified text at intermediate steps.
"""
from __future__ import print_function
import os
from parameters import *
from functions import *

directory = os.path.dirname(os.path.realpath(__file__))

filenames = dict()
execfile(os.path.join(directory, 'filenames.sh'), filenames)

intermediate_file = 'delete.me'

with file(filenames['corpus_unmodified']) as f:
    counts = count_words(f)
total_words = sum(counts.values())
print('Total words in corpus : %i' % total_words)

frequent_words = [word for word in counts if counts[word] > experiment_word_occurrence_min]
random.seed(random_seed)
words_experiment_1, words_experiment_2 = [random.sample(frequent_words, number_of_experiment_words) for _ in range(2)]

with file(filenames['word_freq_experiment_words'], 'w') as f:
    for word in words_experiment_1:
        print('%s,%i' % (word, counts[word]), file=f)

with file(filenames['coocc_noise_experiment_words'], 'w') as f:
    for word in words_experiment_2:
        print('%s,%i' % (word, counts[word]), file=f)

# perform the replacement procedures for the word frequency and the noise cooccurrence experiments
word_samplers = {}
for word in words_experiment_1:
    word_samplers[word] = truncated_geometric_sampling(word, word_freq_experiment_ratio, word_freq_experiment_power_max)
for word in words_experiment_2:
    word_samplers[word] = truncated_geometric_sampling(word, coocc_noise_experiment_ratio, coocc_noise_experiment_power_max)

tmp_file = 'delete.me.2'
with open(intermediate_file) as f_in, open(tmp_file, 'w') as f_out:
    replace_words(word_samplers, f_in, f_out)
os.remove(intermediate_file)
intermediate_file = tmp_file

# add noise to the cooccurrence distributions of experiment 2 words
token_freq_dict = dict()
for word in words_experiment_2:
    target_freq = counts[word] * 1. / total_words
    for i in range(1, coocc_noise_experiment_power_max + 1):
        current_freq = target_freq * truncated_geometric_proba(coocc_noise_experiment_ratio, i, coocc_noise_experiment_power_max)
        token_freq_dict[build_experiment_token(word, i)] = target_freq - current_freq

with open(intermediate_file) as f_in, open(filenames['corpus_modified'], 'w') as f_out:
    intersperse_words(token_freq_dict, f_in, f_out)
os.remove(intermediate_file)
