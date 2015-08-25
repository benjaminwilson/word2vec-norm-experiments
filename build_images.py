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
stats.L2_norm_syn0.name = 'L2 norm (syn0)'
stats.L2_norm_syn1neg.name = 'L2 norm (syn1neg)'


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
for word in test_words:
    idxs += wf_experiment_tokens(word)
test_vecs = vectors_syn0.loc[idxs]
cosine_similarity_heatmap(test_vecs, figsize=(12, 10))
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
fig = plt.figure(figsize=(16, 6))
colormap = plt.cm.gist_ncar
colorcycle = [colormap(i) for i in np.linspace(0, 0.9, len(words))]

ax_syn0 = plt.subplot(131)
ax_syn0.set_xlabel('# occurrences')
ax_syn0.set_ylabel('L2 norm')
ax_syn0.set_xscale('log')
ax_syn0.set_color_cycle(colorcycle)
ax_syn0.set_title('syn0', y=1.04)

ax_syn1neg = plt.subplot(132, sharex=ax_syn0, sharey=ax_syn0)
ax_syn1neg.set_xlabel('# occurrences')
ax_syn1neg.set_color_cycle(colorcycle)
ax_syn1neg.set_title('syn1neg', y=1.04)
    
def plot_for_word(ax, word, series, **kwargs):
    idxs = [build_experiment_token(word, i) for i in range(1, max(word_freq_experiment_ratio, word_freq_experiment_power_max) + 1)]
    x = stats.loc[idxs].occurrences
    y = series.loc[idxs]
    return ax.plot(x, y, marker='o', **kwargs)[0]

lines = []
for word in words:
    lines.append(plot_for_word(ax_syn0, word, stats.L2_norm_syn0))

for word in words:
    plot_for_word(ax_syn1neg, word, stats.L2_norm_syn1neg)

ax_syn0.set_ylim(0, 45)

_ = fig.legend(lines, words, bbox_to_anchor=(0.76, 0.56), loc='center', fontsize=14, frameon=False)
plt.tight_layout()

plt.savefig('outputs/word-frequency-experiment-graph.eps')


## CO-OCCURRENCE NOISE EXPERIMENT

words = coocc_noise_experiment_words
fig = plt.figure(figsize=(16, 6))
colormap = plt.cm.gist_ncar
colorcycle = [colormap(i) for i in np.linspace(0, 0.9, len(words))]

ax_syn0 = plt.subplot(131)
ax_syn0.set_xlabel('Proportion of noise occurrences')
ax_syn0.set_ylabel('L2 norm')
ax_syn0.set_color_cycle(colorcycle)
ax_syn0.set_title('syn0', y=1.04)

ax_syn1neg = plt.subplot(132, sharex=ax_syn0, sharey=ax_syn0)
ax_syn1neg.set_xlabel('Proportion of noise occurrences')
ax_syn1neg.set_color_cycle(colorcycle)
ax_syn1neg.set_title('syn1neg', y=1.04)
    
def plot_for_word(ax, word, series, **kwargs):
    exponents = filter(lambda i: build_experiment_token(word, i) in stats.index, range(1, coocc_noise_experiment_power_max + 1))
    idxs = [build_experiment_token(word, i) for i in exponents]
    x = [1 - coocc_noise_experiment_ratio ** exponent for exponent in exponents]
    y = series.loc[idxs]
    return ax.plot(x, y, marker='o', **kwargs)[0]

lines = []
for word in words:
    lines.append(plot_for_word(ax_syn0, word, stats.L2_norm_syn0))

for word in words:
    plot_for_word(ax_syn1neg, word, stats.L2_norm_syn1neg)

ax_syn0.set_ylim(0, 12)
ax_syn0.set_xlim(0, 1)

_ = fig.legend(lines, words, bbox_to_anchor=(0.76, 0.56), loc='center', fontsize=14, frameon=False)
plt.tight_layout()

plt.savefig('outputs/cooccurrence-noise-graph.eps')


idxs = []
for word in random.sample(coocc_noise_experiment_words, 4):
    idxs += [build_experiment_token(word, i) for i in range(1, coocc_noise_experiment_power_max + 1)]

# to check for colinearity, we need to subtract the vector that they all converge to, viz. the meaningless vector.
meaningless_vec = vectors_syn0.loc[build_experiment_token(meaningless_token, 1)]
test_vecs = vectors_syn0.loc[idxs].dropna()
cosine_similarity_heatmap(test_vecs, figsize=(12, 10))
plt.savefig('outputs/cooccurrence-noise-heatmap.eps')
