"""
Modifies an input text for the word frequency experiment according to the
parameters defined in parameters.py
Reads from stdin, writes to stdout.
Requires sufficient diskspace to write out the modified text at intermediate
steps.
"""
import os
import sys
from parameters import *
from functions import *

wf_experiment_words = read_words(sys.argv[1])
counts = read_word_counts(sys.argv[2])
total_words = sum(counts.values())

# intersperse the meaningless token throughout the corpus
intermediate_file = 'delete.me.word_freq_experiment'
with open(intermediate_file, 'w') as f_out:
    intersperse_words({meaningless_token: meaningless_token_frequency}, sys.stdin, f_out)
wf_experiment_words.append(meaningless_token)

# perform the replacement procedure
word_samplers = {}
distn = lambda i: truncated_geometric_proba(word_freq_experiment_ratio, i, word_freq_experiment_power_max)
for word in wf_experiment_words:
    word_samplers[word] = distribution_to_sampling_function(word, distn, word_freq_experiment_power_max)

with open(intermediate_file) as f_in:
    replace_words(word_samplers, f_in, sys.stdout)
os.remove(intermediate_file)
