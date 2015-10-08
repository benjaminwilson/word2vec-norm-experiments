"""
Modifies an input text for the experiments according to the parameters defined in parameters.py
Writes out the words chosen to stdout.
Takes a random seed as the first argument.
"""
import sys
from parameters import *
from functions import *

random.seed(sys.argv[1])
counts = read_word_counts(sys.argv[2])

frequent_words = [word for word in counts if counts[word] > experiment_word_occurrence_min]
experiment_words = random.sample(frequent_words, number_of_experiment_words)

for word in experiment_words:
    print '%s,%i' % (word, counts[word])
