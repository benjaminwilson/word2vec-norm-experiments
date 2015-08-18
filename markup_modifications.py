"""
Reads a sample text from stdin and writes to stdout LaTeX figures, with
highlighting, showing the corpus modifications made for the two experiments
with the modifications highlighted in red.
""" #FIXME
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
word_samplers = {word: truncated_geometric_sampling(word, word_freq_experiment_ratio, word_freq_experiment_power_max)}

tmp_file = StringIO()
with file('article/sample_input') as f_in:
    replace_words(word_samplers, f_in, tmp_file)

with file('article/word-frequency-experiment-text-cat.tex', 'w') as f_out:
    write_markup(tmp_file.getvalue(), f_out)

# PERFORM THE REPLACEMENT PROCEDURE FOR THE WORD FREQUENCY EXPERIMENT FOR THE MEANINGLESS WORD

word_samplers = {meaningless_token: truncated_geometric_sampling(word, word_freq_experiment_ratio, word_freq_experiment_power_max)}

# intersperse the meaningless token
tmp_file1 = StringIO()
with file('article/sample_input') as f_in:
    intersperse_words({meaningless_token: meaningless_token_frequency}, f_in, tmp_file1)
tmp_file1.seek(0)

tmp_file2 = StringIO()
replace_words(word_samplers, tmp_file1, tmp_file2)

with file('article/word-frequency-experiment-text-void.tex', 'w') as f_out:
    write_markup(tmp_file2.getvalue(), f_out)

# COOCCURRENCE NOISE EXPERIMENT
coocc_noise_experiment_power_max = 3 # better for the example, be sure to note in caption
word = 'cat'
word_samplers = {word: truncated_geometric_sampling(word, coocc_noise_experiment_ratio, coocc_noise_experiment_power_max)}

with file('article/sample_input') as f_in:
    counts = count_words(f_in)
total_words = sum(counts.values())

tmp_file1 = StringIO()
with file('article/sample_input') as f_in:
    replace_words(word_samplers, f_in, tmp_file1)
tmp_file1.seek(0)

# add noise to the cooccurrence distributions of experiment 2 words
token_freq_dict = dict()
target_freq = counts[word] * 1. / total_words
for i in range(1, coocc_noise_experiment_power_max + 1):
    current_freq = target_freq * truncated_geometric_proba(coocc_noise_experiment_ratio, i, coocc_noise_experiment_power_max)
    token_freq_dict[build_experiment_token(word, i)] = target_freq - current_freq

tmp_file2 = StringIO()
intersperse_words(token_freq_dict, tmp_file1, tmp_file2)

with file('article/cooccurrence-noise-experiment.tex', 'w') as f_out:
    write_markup(tmp_file2.getvalue(), f_out)
