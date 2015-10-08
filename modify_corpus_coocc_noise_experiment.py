"""
Modifies an input text for the cooccurrence noise variation experiment
according to the parameters defined in parameters.py
Reads from stdin, writes to stdout.
Requires sufficient diskspace to write out the modified text at intermediate
steps.
"""
import os
import sys
from parameters import *
from functions import *

cn_experiment_words = read_words(sys.argv[1])
counts = read_word_counts(sys.argv[2])
total_words = sum(counts.values())

# perform the replacement procedure
word_samplers = {}
distn = lambda i: evenly_spaced_proba(i, coocc_noise_experiment_max_value)
for word in cn_experiment_words:
    word_samplers[word] = distribution_to_sampling_function(word, distn, coocc_noise_experiment_max_value)

intermediate_file = 'delete.me.coocc_noise_experiment'
with open(intermediate_file, 'w') as f_out:
    replace_words(word_samplers, sys.stdin, f_out)

# add noise to the cooccurrence distributions
token_freq_dict = dict()
for word in cn_experiment_words:
    original_freq = counts[word] * 1. / total_words
    target_freq = original_freq * coocc_noise_experiment_freq_reduction
    for i in range(1, coocc_noise_experiment_max_value + 1):
        current_freq = original_freq * distn(i)
        token_freq_dict[build_experiment_token(word, i)] = target_freq - current_freq

with open(intermediate_file) as f_in:
    intersperse_words(token_freq_dict, f_in, sys.stdout)
os.remove(intermediate_file)
