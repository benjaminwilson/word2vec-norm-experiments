import numpy as np
import pandas as pd
import sys

from parameters import *
from functions import *


word_counts_filename = sys.argv[1]
with file(word_counts_filename) as f:
    word_counts = read_word_counts(f)

words = word_counts.keys()
stats = pd.DataFrame({'occurrences': [word_counts[word] for word in words], 'word': words})
stats['occurrences_band'] = np.floor(np.log2(stats.occurrences)).astype(int)
df = stats.groupby('occurrences_band').word.aggregate({'number': len, 'words': lambda words: ', '.join(random.sample(words, min(len(words), 4)))}).sort_index()

print r'\begin{tabular}{c | r | l}'
print r'frequency band & \# words & example words  \\'
print '\hline'
for band, row in df.iterrows():
    print r'$2^{%i} - 2^{%i}$ & %i & \word{%s} \\' % (band, band+1, row['number'], row['words'])
print r'\end{tabular}'


