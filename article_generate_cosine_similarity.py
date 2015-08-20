import numpy as np
import pandas as pd
import sys

from parameters import *
from functions import *


vectors_syn0_filename = sys.argv[1]
word = sys.argv[2] # e.g. 'the'

vectors, norms = row_normalise_dataframe(load_word2vec_binary(vectors_syn0_filename))

# e.g. 'THE_1' ...
tokens = [build_experiment_token(word, i) for i in range(1, max(word_freq_experiment_ratio, word_freq_experiment_power_max) + 1)]
tokens = [idx for idx in tokens if idx in vectors.index]

non_experiment_idxs = [idx for idx in vectors.index if idx != idx.upper()] # all non uppercase words are non experiment words
non_experiment_vectors = vectors.loc[non_experiment_idxs]

print r'\begin{tabular}{l | c | l}'
print r'token & similarity to \word{%s} & most similar words in unmodified corpus\\' % tokens[0].replace('_', '\_')
print '\hline'
for token in tokens:
    cs = vectors.loc[token].dot(vectors.loc[tokens[0]])
    bydist = by_distance_from(non_experiment_vectors, vectors.loc[token])
    similiar_words = ', '.join(bydist[0:4].index)
    print r'\word{%s} & %.4f & \word{%s} \\' % (token.replace('_', '\_'), cs, similiar_words)
print r'\end{tabular}'


