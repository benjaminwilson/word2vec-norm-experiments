"""
A simple script for outputing the frequencies of the experiment tokens.
"""
from parameters import *
from functions import *

with file('outputs/word_counts_unmodified_corpus.csv') as f:
    wc_unmodified = read_word_counts(f)
            
with file('outputs/word_counts_modified_corpus.csv') as f:
    wc_modified = read_word_counts(f)

wfve_words = read_words('outputs/word_freq_experiment_words')
cnve_words = read_words('outputs/coocc_noise_experiment_words')

print 'word freq variation experiment token frequencies:'
for word in wfve_words:
    print 'token "%s" with original frequency %i' % (word, wc_unmodified[word])
    for index in range(1, word_freq_experiment_power_max + 1):
        token = build_experiment_token(word, index)
        freq = wc_modified[token] if token in wc_modified else 0
        print token, freq
    print '-' * 80
print '=' * 80

print 'coocc noise variation experiment token frequencies:'
for word in cnve_words:
    print 'token "%s" with original frequency %i' % (word, wc_unmodified[word])
    for index in range(1, coocc_noise_experiment_max_value + 1):
        token = build_experiment_token(word, index)
        freq = wc_modified[token] if token in wc_modified else 0
        print token, freq
    print '-' * 80
print '=' * 80
