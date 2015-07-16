"""
Reads a sample text from stdin and writes to stdout LaTeX figures, with
highlighting, showing the corpus modifications made for the two experiments
with the modifications highlighted in red.
"""
import sys
import re
from StringIO import StringIO
from parameters import *
from functions import *

def markup(raw_output):
    hl = re.sub(r'([A-Z]+_[0-9]+)', r'\\textcolor{red}{\1}', raw_output)
    return re.sub('_', '\_', hl)

input_text = sys.stdin.read()
counts = count_words(StringIO(input_text))
total_words = sum(counts.values())

# intersperse the meaningless token throughout the corpus
w_meaningless = StringIO()
intersperse_words({meaningless_token: meaningless_token_frequency}, StringIO(input_text), w_meaningless)

# perform the replacement procedures for the word frequency experiment
word_samplers = {}
for word in ['cat', meaningless_token]:
    word_samplers[word] = truncated_geometric_sampling(word, word_freq_experiment_ratio, word_freq_experiment_power_max)

w_replacement = StringIO()
replace_words(word_samplers, StringIO(w_meaningless.getvalue()), w_replacement)

print r"""\begin{figure}[t]
	\texttt{"""
print markup(w_replacement.getvalue()).strip()
print r"""}
\caption{Text modified for the word frequency experiment, where the word \word{cat} was chosen, $\lambda=0.5$ and $n=20$.}
\label{fig:word-frequency-experiment-text}
\end{figure}"""

print '-' * 80

# cooccurrence noise experiment
coocc_noise_experiment_power_max = 3 # better for the example, be sure to note in caption
word = 'cat'
word_samplers = {}
word_samplers[word] = truncated_geometric_sampling(word, coocc_noise_experiment_ratio, coocc_noise_experiment_power_max)

w_replacement = StringIO()
replace_words(word_samplers, StringIO(input_text), w_replacement)

# add noise to the cooccurrence distributions of experiment 2 words
token_freq_dict = dict()
target_freq = counts[word] * 1. / total_words
for i in range(1, coocc_noise_experiment_power_max + 1):
    current_freq = target_freq * truncated_geometric_proba(coocc_noise_experiment_ratio, i, coocc_noise_experiment_power_max)
    token_freq_dict[build_experiment_token(word, i)] = target_freq - current_freq

w_noise = StringIO()
intersperse_words(token_freq_dict, StringIO(w_replacement.getvalue()), w_noise)

print r"""\begin{figure}[t]
	\texttt{"""
print markup(w_noise.getvalue()).strip()
print r"""}
\caption{Text modified for the co-occurrence noise experiment, where the word \word{cat} was chosen, $\lambda = 5/6$ and $n=3$.}
\label{fig:cooccurrence-noise-experiment-text}
\end{figure}"""
