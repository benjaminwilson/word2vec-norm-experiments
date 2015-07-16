HELP_STR = """
Cleans a two column CSV (id, text) read from stdin, removing punctuation, optionally
lowercasing, and writing out clean UTF-8 text.  Decoding errors are ignored and
lines whose cleaning results in other exceptions are skipped. Unescapes \\n linebreaks."""

import pandas as pd
import argparse
import sys
import re

"""
words consist of 1 or more alphanumeric characters where there first is not
numeric and not an underscore. alphanumeric is defined in the unicode sense.
includes also all of greek, arabic, etc.
"""
TOKEN_PATTERN = r'(?u)\b[^(\W|\d|_)]{1,}\w+\b'
ESCAPED_SEQ = r'\n'
UNESCAPE_PROG = re.compile(re.escape(ESCAPED_SEQ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=HELP_STR)
    parser.add_argument(
        '--case-sensitive', action='store_true')
    args = parser.parse_args()

    tokenizer = re.compile(TOKEN_PATTERN)
    unescape_line_breaks = lambda unicode_text: UNESCAPE_PROG.sub(u'\n', unicode_text)


    doc_chunks = pd.read_csv(sys.stdin,
                             header=None,
                             index_col=0,
                             squeeze=True,
                             chunksize=1000)
    for chunk in doc_chunks:
        chunk = chunk.str.decode('utf-8').fillna(u'')
        chunk = chunk.apply(unescape_line_breaks)
        for _, text in chunk.iteritems():
            try:
                if not args.case_sensitive:
                    text = text.lower()
                tokens = tokenizer.findall(text)
                cleaned = u' '.join(tokens)
                print cleaned.encode('utf-8')
            except ValueError as e:
                pass
