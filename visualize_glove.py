import numpy as np

# Get the interactive Tools for Matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.decomposition import PCA

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import os 
# This class generates a PCA scatterplot of the GloVe vectors we generated for a sanity check using Gensim
# Used some code from https://web.stanford.edu/class/cs224n/materials/Gensim%20word%20vector%20visualization.html
dir_path = os.path.dirname(os.path.realpath(__file__))
glove_file = datapath(dir_path + '/vectors.txt')
word2vec_glove_file = get_tmpfile("vectors.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

top_200_words = []
f = open(dir_path + '/vocab.txt', 'r')
for i in range(200):
    top_200_words.append(f.readline().split(" ")[0])

def display_pca_scatterplot(model, words):
    word_vectors = np.array([model[w] for w in words])
    twodim = PCA().fit_transform(word_vectors)[:,:2]
    
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, twodim):
        plt.text(x + 0.05, y + 0.05, word)
    plt.show()

display_pca_scatterplot(model, top_200_words)
