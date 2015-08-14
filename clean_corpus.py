import sys
import csv
import re
csv.field_size_limit(sys.maxsize)

"""
words consist of 1 or more alphanumeric characters where there first is not
numeric and not an underscore. alphanumeric is defined in the unicode sense.
includes also all of greek, arabic, etc.
"""
TOKEN_PATTERN = r'(?u)\b[^(\W|\d|_)]{1,}\w+\b'

tokenizer = re.compile(TOKEN_PATTERN)
reader = csv.reader(sys.stdin, delimiter=',', quotechar='"')
for text_id, text in reader:
    text = text.decode('utf-8').lower()
    tokens = tokenizer.findall(text)
    cleaned = u' '.join(tokens)
    print cleaned.encode('utf-8')
