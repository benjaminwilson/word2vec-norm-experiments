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

.PHONY: clean experiment
experiment: $(vectors_binary_syn0) $(vectors_binary_syn1neg)
clean:
	rm $(corpus_modified)
	rm $(vectors_binary_syn0)
	rm $(vectors_binary_syn1neg)

$(corpus_unmodified):
	wget -qO- http://lateral-datadumps.s3-website-eu-west-1.amazonaws.com/wikipedia_utf8_filtered_20pageviews.csv.gz \
		| gunzip -c \
		| python clean_corpus.py \
		> $(corpus_unmodified)
$(word_counts_unmodified_corpus): $(corpus_unmodified)
	python count_words.py > $(word_counts_unmodified_corpus) < $(corpus_unmodified)	
$(word_freq_experiment_words): $(word_counts_unmodified_corpus)
	python choose_experiment_words.py randomseed1 > $(word_freq_experiment_words) < $(word_counts_unmodified_corpus)
	cat $(word_counts_unmodified_corpus) | grep ^the, >> $(word_freq_experiment_words)
$(coocc_noise_experiment_words): $(word_counts_unmodified_corpus)
	python choose_experiment_words.py randomseed2 > $(coocc_noise_experiment_words) < $(word_counts_unmodified_corpus)
$(corpus_modified): $(corpus_unmodified) $(word_counts_unmodified_corpus) $(word_freq_experiment_words) $(coocc_noise_experiment_words)
	cat $(corpus_unmodified) \
		| python modify_corpus_word_freq_experiment.py $(word_freq_experiment_words) $(word_counts_unmodified_corpus) \
		| python modify_corpus_coocc_noise_experiment.py $(coocc_noise_experiment_words) $(word_counts_unmodified_corpus) \
		> $(corpus_modified)
$(word_counts_modified_corpus): $(corpus_modified)
	 python count_words.py > $(word_counts_modified_corpus) < $(corpus_modified)
word2vec: word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
$(vectors_binary_syn0) $(vectors_binary_syn1neg): word2vec $(corpus_modified)
	./word2vec -min-count 128 -hs 0 -negative 5 -window 10 -size 100 -cbow 1 -debug 2 -threads 16 -iter 10 -binary 1 -output $(vectors_binary_syn0) -train $(corpus_modified)

images:
	python build_images.py $(vectors_binary_syn0) $(vectors_binary_syn1neg) $(word_counts_modified_corpus) $(word_freq_experiment_words) $(coocc_noise_experiment_words)
	touch images
article/main.pdf:
	python article_generate_word_counts.py $(word_counts_unmodified_corpus) > article/words-occurrences.tex
	python article_generate_cosine_similarity.py $(vectors_binary_syn0) the > article/cosine-similarity.tex
	python article_markup_wordcounts.py < $(word_freq_experiment_words) > article/word-frequency-experiment-counts.tex
	python article_markup_wordcounts.py < $(coocc_noise_experiment_words) > article/noise-cooccurrence-experiment-counts.tex
	cd article && latex main.tex
	cd article && latex main.tex && dvipdf main.dvi
talk/main.pdf:
	cd talk && pdflatex -shell-escape main.tex
