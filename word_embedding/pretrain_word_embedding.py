import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import collections
import random
import nltk
from scipy import spatial

#Load Glove vectors
LIMIT_VOCAB_SIZE = 40_000
file_name = 'glove.6B.50d.txt'
glove_vocab = []
glove_embed = []
word2idx = {}

PAD_TOKEN = 0
word2idx = {'PAD': PAD_TOKEN}

with (glove_data_directory/file_name).open('r') as file:
    for index, line in enumerate(file):
         values = line.split()
         vocab = values[0]
         glove_vocab.append(vocab)
         word_weight = np.asarray(values[1:],dtype=np.float32)
         word2idx[word] = index + 1
         glove_embed.append(word_weight)

         if index + 1 == LIMIT_VOCAB_SIZE:
             #Limit vocabulary to top word_size terms
             break

#glove_vocab_size = len(glove_vocab)
EMBEDDING_DIMEMSION = len(glove_embed[0])
glove_embed.insert(0, np.random.randn(EMBEDDING_DIMEMSION))

UNKNOWN_TOKEN = len(glove_embed)
word2idx['UNK'] = UNKNOWN_TOKEN
glove_embed.append(np.random.randn(EMBEDDING_DIMEMSION))

glove_embed = np.asarray(glove_embed, dtype = np.float32)

VOCAB_SIZE = weights.shape[0]

# Embedding in Fensorflow
feature = {}
features['word_indices'] = nltk.word_tokenize('hello world')
features['word_indices'] = [word2idx.get(word, UNKNOWN_TOKEN) for word in features['word_indices']]
glove_weights_initializer = tf.constant_initializer(glove_embed)
embedding_weights = tf.get_variable(
    name = 'embedding_weights',
    shape = (VOCAB_SIZE, EMBEDDING_DIMEMSION),
    initializer = glove_weights_initializer,
    trainable= False)

embedding = tf.nn.embedding_lookup(embedding_weights, features['word_indices'])
