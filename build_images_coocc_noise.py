"""
Given the vectors and word counts resulting from the experiments, build
graphics for the cooccurrence noise variation experiment section of the
article.
"""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg') # must be set before importing pyplot
import matplotlib.pyplot as plt
import random
import sys

from parameters import *
from functions import *

random.seed(1) # fix seed so that we can refer to the randomly chosen words in the article body

vectors_syn0_filename = sys.argv[1]
vectors_syn1neg_filename = sys.argv[2]
word_counts_filename = sys.argv[3]
coocc_noise_exp_words_filename = sys.argv[4]

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'figure.autholayout': True})

coocc_noise_experiment_words = read_words(coocc_noise_exp_words_filename)

def calculate_norms(vecs):
    return np.sqrt((vecs ** 2).sum(axis=1))

vectors_syn0 = load_word2vec_binary(vectors_syn0_filename)
norms_syn0 = calculate_norms(vectors_syn0)

vectors_syn1neg = load_word2vec_binary(vectors_syn1neg_filename)
norms_syn1neg = calculate_norms(vectors_syn1neg)

vocab = list(vectors_syn0.index)
counts_modified_corpus = read_word_counts(word_counts_filename)

stats = pd.DataFrame({'occurrences': counts_modified_corpus,
                      'L2_norm_syn0': norms_syn0,
                      'L2_norm_syn1neg': norms_syn1neg}).dropna()

# WORD VECTOR LENGTH AS A FUNCTION OF NOISE PROPORTION

# reorder by the left-most value of the corresponding syn0 plot
# i.e. by word vector length of the noiseless token
words = sorted(coocc_noise_experiment_words,
               key=lambda word: stats.L2_norm_syn0.loc[build_experiment_token(word, 1)],
               reverse=True)

markers = [['o', 's', 'D'][i % 3] for i in range(len(words))]
fig = plt.figure(figsize=(16, 6))
colorcycle = plt.cm.gist_rainbow(np.linspace(0, 1, 5))

xlabel = 'Proportion of occurrences from noise distribution'
ax_syn0 = plt.subplot(131)
ax_syn0.set_xlabel(xlabel)
ax_syn0.set_ylabel('vector length')
ax_syn0.set_color_cycle(colorcycle)
ax_syn0.set_title('syn0', y=1.04)

ax_syn1neg = plt.subplot(132, sharex=ax_syn0)
ax_syn1neg.set_xlabel(xlabel)
ax_syn1neg.set_color_cycle(colorcycle)
ax_syn1neg.set_title('syn1neg', y=1.04)
    
def plot_for_word(ax, word, series, **kwargs):
    outcomes = range(1, coocc_noise_experiment_max_value + 1)
    x = [noise_proportion(i, coocc_noise_experiment_max_value) for i in outcomes]
    y = series.loc[[build_experiment_token(word, i) for i in outcomes]]
    marker = ['o', 's', 'D'][ord(word[0]) % 3]
    return ax.plot(x, y, marker=marker, **kwargs)[0]

lines = []
for word in words:
    lines.append(plot_for_word(ax_syn0, word, stats.L2_norm_syn0))

for word in words:
    plot_for_word(ax_syn1neg, word, stats.L2_norm_syn1neg)

ax_syn0.set_ylim(0, 30)
ax_syn1neg.set_ylim(0, 15)
ax_syn0.set_xlim(0, 1)

_ = fig.legend(lines, words, bbox_to_anchor=(0.76, 0.56), loc='center', fontsize=14, frameon=False)
plt.tight_layout()

plt.savefig('outputs/cooccurrence-noise-graph.eps')

# COSINE SIMILARITY HEATMAP

idxs = []
ticks = []
for word in random.sample(coocc_noise_experiment_words, 4):
    tokens = [build_experiment_token(word, i) for i in range(1, coocc_noise_experiment_max_value + 1)]
    idxs += tokens
    ticks += tokens[:2] + ['.  '] * len(tokens[2:-1]) + tokens[-1:]

example_vecs = vectors_syn0.loc[idxs]
cosine_similarity_heatmap(example_vecs, ticks, figsize=(12, 10))
plt.savefig('outputs/cooccurrence-noise-heatmap.eps')
