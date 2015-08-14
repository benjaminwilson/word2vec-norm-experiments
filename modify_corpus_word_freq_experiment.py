"""
Modifies an input text for the word frequency experiment.
Reads from stdin, writes to stdout.
Requires sufficient diskspace to write out the modified text at intermediate steps.
"""
import os
import sys
from parameters import *
from functions import *

wf_experiment_words = read_words(sys.argv[1])
with file(sys.argv[2]) as f:
    counts = read_word_counts(f)
total_words = sum(counts.values())

# intersperse the meaningless token throughout the corpus
intermediate_file = 'delete.me'
with open(intermediate_file, 'w') as f_out:
    intersperse_words({meaningless_token: meaningless_token_frequency}, sys.stdin, f_out)
wf_experiment_words.append(meaningless_token)

# perform the replacement procedure
word_samplers = {}
for word in wf_experiment_words:
    word_samplers[word] = truncated_geometric_sampling(word, word_freq_experiment_ratio, word_freq_experiment_power_max)

with open(intermediate_file) as f_in:
    replace_words(word_samplers, f_in, sys.stdout)
os.remove(intermediate_file)
