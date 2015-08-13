import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg') # must be set before importing pyplot
import matplotlib.pyplot as plt
import random

from parameters import *
from functions import *

filenames = dict()
execfile('filenames.sh', filenames)

matplotlib.rcParams.update({'font.size': 14})

word_freq_experiment_words = read_words(filenames['word_freq_experiment_words'])
coocc_noise_experiment_words = read_words(filenames['coocc_noise_experiment_words'])

vectors_syn0 = load_word2vec_binary(filenames['vectors_binary'])
norms_syn0 = np.sqrt((vectors_syn0 ** 2).sum(axis=1))

vectors_syn1neg = load_word2vec_binary(filenames['vectors_binary'] + '.syn1neg')
norms_syn1neg = np.sqrt((vectors_syn1neg ** 2).sum(axis=1))

vocab = list(vectors_syn0.index)

# Calculate frequencies in the modified corpus
with file(filenames['word_counts_modified_corpus']) as f:
    new_counts = read_word_counts(f)
total_words = sum(new_counts.values())

stats = pd.DataFrame({'occurrences': new_counts, 'L2_norm_syn0': norms_syn0, 'L2_norm_syn1neg': norms_syn1neg}).dropna()
stats['occurrences_band'] = np.floor(np.log2(stats.occurrences)).astype(int)
stats['log2_frequency'] = np.log2(stats.occurrences * 1. / total_words)
stats.L2_norm_syn0.name = 'L2 norm (syn0)'
stats.L2_norm_syn1neg.name = 'L2 norm (syn1neg)'

band_counts = stats.occurrences_band.dropna().value_counts().sort_index()
band_counts.index.name = 'log2 (# occurrences)'

_ = band_counts.plot(kind='bar')
plt.title('Number of words in each occurrence count band', y=1.08)
plt.tight_layout()
plt.savefig('outputs/occurrence-histogram.eps')

band_gb = stats.groupby('occurrences_band')
means = band_gb.L2_norm_syn0.mean()
errors = band_gb.L2_norm_syn0.std()
ax = means.plot(yerr=errors)

means = band_gb.L2_norm_syn1neg.mean()
errors = band_gb.L2_norm_syn1neg.std()
ax = means.plot(yerr=errors, figsize=(7,7))

_ = ax.set_xlim(7, 20)
_ = ax.set_ylim(0, 60)

_ = ax.set_xlabel('log2 (# occurrences)')
_ = ax.set_ylabel('L2 norm')

plt.title('Mean and Std. of norm as function of # occurrences', y=1.04)
plt.legend(['syn0 vectors', 'syn1neg vectors'])
plt.savefig('outputs/frequency-norm-graph.eps')

# scatter plot of syn0 norm vs frequency for a sample of ordinary (non-experiment-) words
fig = plt.figure(figsize=(16, 8))
non_experiment_words = [word for word in stats.index if word != word.upper()]
sample = stats.loc[random.sample(non_experiment_words, 15000)]
plt.scatter(sample.occurrences, sample.L2_norm_syn0, s=0.2)
plt.xscale('log')
plt.ylim(0, 50)
plt.xlim(200, 10 ** 7)
plt.title('Frequency vs vector length', y=1.04).set_fontsize(30)
plt.xlabel('# occurrences').set_fontsize(30)
plt.ylabel('L2 norm (syn0)').set_fontsize(30)
plt.savefig('outputs/frequency-norm-scatterplot.eps')


def set_num_plots(num_plots):
    colormap = plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

words = word_freq_experiment_words
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
set_num_plots(len(words))
for word in words:
    lines.append(plot_for_word(ax_syn0, word, stats.L2_norm_syn0))

set_num_plots(len(words))
for word in words:
    plot_for_word(ax_syn1neg, word, stats.L2_norm_syn1neg)

ax_syn0.set_ylim(0, 35)

_ = fig.legend(lines, words, bbox_to_anchor=(0.76, 0.56), loc='center', fontsize=14, frameon=False)
plt.tight_layout()

plt.savefig('outputs/word-frequency-experiment-graph.eps')


test_words = random.sample(word_freq_experiment_words, 4) + ['the', meaningless_token]
idxs = [build_experiment_token(word, i) for word in test_words for i in range(1, max(word_freq_experiment_ratio, word_freq_experiment_power_max) + 1)]
test_vecs = vectors_syn0.loc[idxs].dropna()
cosine_similarity_heatmap(test_vecs, figsize=(12, 10))
plt.savefig('outputs/word-frequency-experiment-heatmap.eps')


## Co-occurrence noise experiment

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
set_num_plots(len(words))
for word in words:
    lines.append(plot_for_word(ax_syn0, word, stats.L2_norm_syn0))

set_num_plots(len(words))
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
test_vecs = vectors_syn0.loc[idxs].dropna() - meaningless_vec
cosine_similarity_heatmap(test_vecs, figsize=(12, 10))
plt.savefig('outputs/cooccurrence-noise-heatmap.eps')