import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg') # must be set before importing pyplot
import matplotlib.pyplot as plt
import random
import sys

from parameters import *
from functions import *

random.seed(0) # fix seed so that we can refer to the randomly chosen words in the article body

vectors_syn0_filename = sys.argv[1]
vectors_syn1neg_filename = sys.argv[2]
word_counts_filename = sys.argv[3]
word_freq_exp_words_filename = sys.argv[4]
coocc_noise_exp_words_filename = sys.argv[5]

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'figure.autholayout': True})

word_freq_experiment_words = read_words(word_freq_exp_words_filename)
coocc_noise_experiment_words = read_words(coocc_noise_exp_words_filename)

vectors_syn0 = load_word2vec_binary(vectors_syn0_filename)
norms_syn0 = np.sqrt((vectors_syn0 ** 2).sum(axis=1))

vectors_syn1neg = load_word2vec_binary(vectors_syn1neg_filename)
norms_syn1neg = np.sqrt((vectors_syn1neg ** 2).sum(axis=1))

vocab = list(vectors_syn0.index)

with file(word_counts_filename) as f:
    new_counts = read_word_counts(f)
total_words = sum(new_counts.values())

stats = pd.DataFrame({'occurrences': new_counts, 'L2_norm_syn0': norms_syn0, 'L2_norm_syn1neg': norms_syn1neg}).dropna()
stats['occurrences_band'] = np.floor(np.log2(stats.occurrences)).astype(int)
stats['log2_frequency'] = np.log2(stats.occurrences * 1. / total_words)
stats.L2_norm_syn0.name = 'vector length (syn0)'
stats.L2_norm_syn1neg.name = 'vector length (syn1neg)'


# WORD FREQUENCY EXPERIMENT

def wf_experiment_tokens(word):
    """
    return the list of tokens introduced into the corpus in the word frequency
    experiment for the given word.
    """
    idxs = [build_experiment_token(word, i) for i in range(1, max(word_freq_experiment_ratio, word_freq_experiment_power_max) + 1)]
    return [idx for idx in idxs if idx in vectors_syn0.index]

# cosine similarity heatmap
test_words = random.sample([word for word in word_freq_experiment_words if word != 'the'], 4)
test_words += ['the', meaningless_token]
idxs = []
ticks = []
for word in test_words:
    tokens = wf_experiment_tokens(word) 
    idxs += tokens
    ticks += tokens[:2] + ['.  '] * len(tokens[2:-1]) + tokens[-1:]

test_vecs = vectors_syn0.loc[idxs]
cosine_similarity_heatmap(test_vecs, ticks, figsize=(12, 10))
plt.savefig('outputs/word-frequency-experiment-heatmap.eps')

# remove word vectors that are not well trained
# drop those whose cosine similarity with word_1 is < 0.8
for word in word_freq_experiment_words + [meaningless_token]:
    idxs = wf_experiment_tokens(word) # ordered: word_1, word_2 ..
    test_vecs = vectors_syn0.loc[idxs]
    cs = cosine_similarity(test_vecs)
    poorly_trained = [idx for i, idx in enumerate(idxs) if cs[0,i] < 0.8]
    stats.drop(poorly_trained, axis=0, inplace=True)
    vectors_syn0.drop(poorly_trained, axis=0, inplace=True)
    vectors_syn1neg.drop(poorly_trained, axis=0, inplace=True)


words = word_freq_experiment_words + [meaningless_token]

# reorder by the left-most value of the corresponding syn0 plot
def get_leftmost_value(word):
    for i in range(1, word_freq_experiment_power_max + 1):
        idx = build_experiment_token(word, i)
        if idx in vectors_syn0.index:
            last_idx = idx
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
    
def plot_for_word(ax, word, marker, series, **kwargs):
    idxs = [build_experiment_token(word, i) for i in range(1, word_freq_experiment_power_max + 1)]
    x = stats.loc[idxs].occurrences.dropna()
    y = series.loc[idxs].dropna()
    return ax.plot(x, y, marker=marker, **kwargs)[0]

lines = []
for word, marker in zip(words, markers):
    lines.append(plot_for_word(ax_syn0, word, marker, stats.L2_norm_syn0))

for word, marker in zip(words, markers):
    plot_for_word(ax_syn1neg, word, marker, stats.L2_norm_syn1neg)

ax_syn0.set_ylim(0, 45)
ax_syn1neg.set_ylim(0, 25)

_ = fig.legend(lines, words, bbox_to_anchor=(0.76, 0.56), loc='center', fontsize=14, frameon=False)
plt.tight_layout()

plt.savefig('outputs/word-frequency-experiment-graph.eps')


## CO-OCCURRENCE NOISE EXPERIMENT

words = coocc_noise_experiment_words

# reorder by the left-most value of the corresponding syn0 plot
words = sorted(words, key=lambda word: stats.L2_norm_syn0.loc[build_experiment_token(word, 1)], reverse=True)

markers = [['o', 's', 'D'][i % 3] for i in range(len(words))]
fig = plt.figure(figsize=(16, 6))
colorcycle = plt.cm.gist_rainbow(np.linspace(0, 1, 5))

ax_syn0 = plt.subplot(131)
ax_syn0.set_xlabel('Proportion of occurrences from noise distribution')
ax_syn0.set_ylabel('vector length')
ax_syn0.set_color_cycle(colorcycle)
ax_syn0.set_title('syn0', y=1.04)

ax_syn1neg = plt.subplot(132, sharex=ax_syn0, sharey=ax_syn0)
ax_syn1neg.set_xlabel('Proportion of occurrences from noise distribution')
ax_syn1neg.set_color_cycle(colorcycle)
ax_syn1neg.set_title('syn1neg', y=1.04)
    
def plot_for_word(ax, word, series, **kwargs):
    outcomes = range(1, coocc_noise_experiment_max_value + 1)
    idxs = [build_experiment_token(word, i) for i in outcomes]
    x = [noise_proportion(i, coocc_noise_experiment_max_value) for i in outcomes]
    y = series.loc[idxs]
    marker = ['o', 's', 'D'][ord(word[0]) % 3]
    return ax.plot(x, y, marker=marker, **kwargs)[0]

lines = []
for word in words:
    lines.append(plot_for_word(ax_syn0, word, stats.L2_norm_syn0))

for word in words:
    plot_for_word(ax_syn1neg, word, stats.L2_norm_syn1neg)

#ax_syn0.set_ylim(2.5, 8.5)
ax_syn0.set_xlim(0, 1)

_ = fig.legend(lines, words, bbox_to_anchor=(0.76, 0.56), loc='center', fontsize=14, frameon=False)
plt.tight_layout()

plt.savefig('outputs/cooccurrence-noise-graph.eps')


idxs = []
ticks = []
for word in random.sample(coocc_noise_experiment_words, 4):
    candidates = [build_experiment_token(word, i) for i in range(1, coocc_noise_experiment_max_value + 1)]
    tokens = [token for token in candidates if token in vectors_syn0.index]
    idxs += tokens
    ticks += tokens[:2] + ['.  '] * len(tokens[2:-1]) + tokens[-1:]

test_vecs = vectors_syn0.loc[idxs]
cosine_similarity_heatmap(test_vecs, ticks, figsize=(12, 10))
plt.savefig('outputs/cooccurrence-noise-heatmap.eps')
