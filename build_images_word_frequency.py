"""
Given the vectors and word counts resulting from the experiments, build
graphics for the word frequency variation experiment section of the article.
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
word_freq_exp_words_filename = sys.argv[4]

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'figure.autholayout': True})

word_freq_experiment_words = read_words(word_freq_exp_words_filename)

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

def wf_experiment_tokens(word):
    """
    return the list of tokens introduced into the corpus in the word frequency
    experiment for the given word for which we have word vectors.
    """
    idxs = [build_experiment_token(word, i) for i in range(1, word_freq_experiment_power_max + 1)]
    return [idx for idx in idxs if idx in vectors_syn0.index]

# COSINE SIMILARITY HEATMAP
example_words = random.sample([word for word in word_freq_experiment_words if word != 'the'], 4)
example_words += ['the', meaningless_token]
idxs = []
ticks = []
for word in example_words:
    tokens = wf_experiment_tokens(word) 
    idxs += tokens
    ticks += tokens[:2] + ['.  '] * len(tokens[2:-1]) + tokens[-1:]

example_vecs = vectors_syn0.loc[idxs]
cosine_similarity_heatmap(example_vecs, ticks, figsize=(12, 10))
plt.savefig('outputs/word-frequency-experiment-heatmap.eps')

# for the experiment words, drop those word vectors that are not well trained
# specifically, those whose cosine similarity with word_1 is < 0.8
for word in word_freq_experiment_words + [meaningless_token]:
    idxs = wf_experiment_tokens(word) # ordered: word_1, word_2 ..
    example_vecs = vectors_syn0.loc[idxs]
    cs = cosine_similarity(example_vecs)
    poorly_trained = [idx for i, idx in enumerate(idxs) if cs[0,i] < 0.8]
    stats.drop(poorly_trained, axis=0, inplace=True)
    vectors_syn0.drop(poorly_trained, axis=0, inplace=True)
    vectors_syn1neg.drop(poorly_trained, axis=0, inplace=True)


# GRAPH OF THE WORD VECTOR LENGTHS
words = word_freq_experiment_words + [meaningless_token]

# reorder by the left-most value of the corresponding syn0 plot
# i.e. by word vector length of the lowest frequency token for each word
def get_leftmost_value(word):
    last_idx = wf_experiment_tokens(word)[-1]
    return stats.L2_norm_syn0.loc[last_idx]
words = sorted(words, key=get_leftmost_value, reverse=True)
markers = [['o', 's', 'D'][i % 3] for i in range(len(words))]

fig = plt.figure(figsize=(16, 6))
colorcycle = plt.cm.gist_rainbow(np.linspace(0, 1, 5))

ax_syn0 = plt.subplot(131)
ax_syn0.set_xlabel('frequency')
ax_syn0.set_ylabel('vector length')
ax_syn0.set_xscale('log')
ax_syn0.set_color_cycle(colorcycle)
ax_syn0.set_title('syn0', y=1.04)

ax_syn1neg = plt.subplot(132, sharex=ax_syn0)
ax_syn1neg.set_xlabel('frequency')
ax_syn1neg.set_color_cycle(colorcycle)
ax_syn1neg.set_title('syn1neg', y=1.04)
    
def plot_for_word(ax, word, marker, column, **kwargs):
    idxs = wf_experiment_tokens(word)
    idxs = [build_experiment_token(word, i) for i in range(1, word_freq_experiment_power_max + 1)]
    x = stats.loc[idxs].occurrences
    y = stats[column].loc[idxs]
    return ax.plot(x, y, marker=marker, **kwargs)[0]

lines = []
for word, marker in zip(words, markers):
    lines.append(plot_for_word(ax_syn0, word, marker, 'L2_norm_syn0'))

for word, marker in zip(words, markers):
    plot_for_word(ax_syn1neg, word, marker, 'L2_norm_syn1neg')

ax_syn0.set_ylim(0, 45)
ax_syn1neg.set_ylim(0, 25)

_ = fig.legend(lines, words, bbox_to_anchor=(0.76, 0.56), loc='center', fontsize=14, frameon=False)
plt.tight_layout()

plt.savefig('outputs/word-frequency-experiment-graph.eps')
