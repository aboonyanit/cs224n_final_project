import numpy as np

# Get the interactive Tools for Matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.decomposition import PCA

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/glove/vectors.txt')
word2vec_glove_file = get_tmpfile("vectors.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)