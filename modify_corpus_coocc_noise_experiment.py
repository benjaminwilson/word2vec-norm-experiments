"""
Modifies an input text for the experiments according to the parameters defined in parameters.py
Reads from stdin, writes to stdout.
Requires sufficient diskspace to write out the modified text at intermediate steps.
"""
import os
import sys
from parameters import *
from functions import *

cn_experiment_words = read_words(sys.argv[1])
with file(sys.argv[2]) as f:
    counts = read_word_counts(f)
total_words = sum(counts.values())

# perform the replacement procedure
word_samplers = {}
for word in cn_experiment_words:
    word_samplers[word] = truncated_geometric_sampling(word, coocc_noise_experiment_ratio, coocc_noise_experiment_power_max)

intermediate_file = 'delete.me'
with open(intermediate_file, 'w') as f_out:
    replace_words(word_samplers, sys.stdin, f_out)

# add noise to the cooccurrence distributions
token_freq_dict = dict()
for word in cn_experiment_words:
    target_freq = counts[word] * 1. / total_words
    for i in range(1, coocc_noise_experiment_power_max + 1):
        current_freq = target_freq * truncated_geometric_proba(coocc_noise_experiment_ratio, i, coocc_noise_experiment_power_max)
        token_freq_dict[build_experiment_token(word, i)] = target_freq - current_freq

with open(intermediate_file) as f_in:
    intersperse_words(token_freq_dict, f_in, sys.stdout)
os.remove(intermediate_file)
