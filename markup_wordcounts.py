import sys

print r"""\begin{center}
\begin{tabular}{c | c}
word & \# occurrences \\
\hline"""
for line in sys.stdin:
    word, count = line.split(',')
    print r'\word{%s} & %s \\' % (word, count.strip())
print r"""\end{tabular}
\end{center}"""

