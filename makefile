include filenames.sh
source_corpus_url="http://lateral-datadumps.s3-website-eu-west-1.amazonaws.com/wikipedia_utf8_filtered_20pageviews.csv.gz"
CC=gcc
CFLAGS=-lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
corpus_unmodified=outputs/corpus-unmodified.txt
corpus_modified=outputs/corpus-modified.txt
word_counts_unmodified_corpus=outputs/word_counts_unmodified_corpus.csv
word_counts_modified_corpus=outputs/word_counts_modified_corpus.csv
vectors_binary_syn0=outputs/vectors.bin
vectors_binary_syn1neg=outputs/vectors.bin.syn1neg
word_freq_experiment_words=outputs/word_freq_experiment_words
coocc_noise_experiment_words=outputs/coocc_noise_experiment_words

all: $(vectors_binary)

$(corpus_unmodified):
	wget -qO- $(source_corpus_url) | gunzip -c | python clean_corpus.py > $(corpus_unmodified)
$(word_counts_unmodified_corpus): $(corpus_unmodified)
	cat $(corpus_unmodified) | python count_words.py > $(word_counts)	
$(word_freq_experiment_words): $(word_counts)
	cat $(word_counts) | python choose_experiment_words.py randomseed1 > $(word_freq_experiment_words)
	echo 'the' >> $(word_freq_experiment_words)
$(coocc_noise_experiment_words): $(word_counts)
	cat $(word_counts) | python choose_experiment_words.py randomseed2 > $(coocc_noise_experiment_words)
$(corpus_modified): $(corpus_unmodified) $(word_counts) $(word_freq_experiment_words) $(coocc_noise_experiment_words)
	python modify_corpus.py
$(word_counts_modified_corpus): $(corpus_modified)
	cat $(corpus_modified) | python count_words.py > $(word_counts_modified_corpus)	
word2vec: word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
$(vectors_binary_syn0) $(vectors_binary_syn1neg): word2vec $(corpus_modified)
	./word2vec -min-count 200 -hs 0 -negative 5 -window 10 -size 100 -cbow 1 -debug 2 -threads 16 -iter 10 -binary 1 -output $(vectors_binary) -train $(corpus_modified)

.PHONY: clean images
images: $(vectors_binary_syn0) $(vectors_binary_syn1neg) $(word_counts_modified_corpus)
	python build_images.py
clean:
	rm $(corpus_modified)
	rm $(vectors_binary_syn0)
	rm $(vectors_binary_syn1neg)