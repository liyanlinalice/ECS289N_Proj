import word2vec

word2vec.word2phrase('./text8', './text8-phrases', verbose=True)
word2vec.word2vec('./text8-phrases', './text8.bin', size=100, verbose=True)