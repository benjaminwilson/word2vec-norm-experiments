"""
Performs the corpus modifications on sample texts for inclusion in the article,
highlighting in red the modifications (in LaTeX).
"""
import re
from StringIO import StringIO
from parameters import *
from functions import *

def write_markup(raw_output, f_out):
    """
    Markup all modifications to a sample text for typesetting in LaTeX.
    """
    f_out.write(r"\texttt{")
    output = re.sub(r'([A-Z]+_[0-9]+)', r'\\textcolor{red}{\1}', raw_output)
    output = re.sub('_', '\_', output)
    output = r'\newline '.join(output.strip().split('\n'))
    f_out.write(output)
    f_out.write(r"}")

# PERFORM THE REPLACEMENT PROCEDURE FOR THE WORD FREQUENCY EXPERIMENT FOR THE WORD "CAT"

word = 'cat'
distn = lambda i: truncated_geometric_proba(word_freq_experiment_ratio, i, word_freq_experiment_power_max)
word_samplers = {word: distribution_to_sampling_function(word, distn, word_freq_experiment_power_max)}

tmp_file = StringIO()
with file('article/word-frequency-experiment-text-cat.input') as f_in:
    replace_words(word_samplers, f_in, tmp_file)

with file('article/word-frequency-experiment-text-cat.tex', 'w') as f_out:
    write_markup(tmp_file.getvalue(), f_out)

# PERFORM THE REPLACEMENT PROCEDURE FOR THE WORD FREQUENCY EXPERIMENT FOR THE MEANINGLESS WORD

# intersperse the meaningless token
tmp_file1 = StringIO()
with file('article/word-frequency-experiment-text-void.input') as f_in:
    intersperse_words({meaningless_token: 0.05}, f_in, tmp_file1)
tmp_file1.seek(0)

tmp_file2 = StringIO()
distn = lambda i: truncated_geometric_proba(word_freq_experiment_ratio, i, word_freq_experiment_power_max)
word_samplers = {meaningless_token: distribution_to_sampling_function(meaningless_token, distn, word_freq_experiment_power_max)}
replace_words(word_samplers, tmp_file1, tmp_file2)

with file('article/word-frequency-experiment-text-void.tex', 'w') as f_out:
    write_markup(tmp_file2.getvalue(), f_out)

# COOCCURRENCE NOISE EXPERIMENT
word = 'cat'
distn = lambda i: evenly_spaced_proba(i, coocc_noise_experiment_max_value)
word_samplers = {word: distribution_to_sampling_function(word, distn, coocc_noise_experiment_max_value)}

with file('article/cooccurrence-noise-experiment.input') as f_in:
    counts = count_words(f_in)
total_words = sum(counts.values())

tmp_file1 = StringIO()
with file('article/cooccurrence-noise-experiment.input') as f_in:
    replace_words(word_samplers, f_in, tmp_file1)
tmp_file1.seek(0)

# add noise to the cooccurrence distribution
token_freq_dict = dict()
target_freq = counts[word] * 1. * coocc_noise_experiment_freq_reduction / total_words
for i in range(1, coocc_noise_experiment_max_value + 1):
    current_freq = counts[word] * evenly_spaced_proba(i, coocc_noise_experiment_max_value) / total_words
    token_freq_dict[build_experiment_token(word, i)] = target_freq - current_freq
tmp_file2 = StringIO()
intersperse_words(token_freq_dict, tmp_file1, tmp_file2)

with file('article/cooccurrence-noise-experiment.tex', 'w') as f_out:
    write_markup(tmp_file2.getvalue(), f_out)
