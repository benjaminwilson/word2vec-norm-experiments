import sys
from functions import *

word_counts = count_words(sys.stdin)
for word, count in word_counts.iteritems():
    print '%s,%i' % (word, count)
