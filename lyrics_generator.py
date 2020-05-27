from __future__ import print_function
# import Keras library
import re

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
import spacy
import numpy as np
import random
import sys
import os
import time
import codecs
import collections
import pickle
from six.moves import cPickle


def load_dictionary(root_path, author_name):
    artist_name_for_document_name = re.sub("[^0-9a-zA-Z]+", "_", author_name)
    dictionary_file_path = root_path + artist_name_for_document_name + "/"

    with open(dictionary_file_path + artist_name_for_document_name + '.pkl', 'rb') as f:
        return pickle.load(f)


def create_wordlist(artist_form_lyrics_folder_path, artist_name):
    loaded_dictionary = load_dictionary(artist_form_lyrics_folder_path, artist_name)
    return list(loaded_dictionary.keys())


'''
MAIN
'''
artist_form_lyrics_folder_path = "./data/rap_lyrics_links/"
artist_name = "Eminem"
save_dir = './words_vocabulary_files/'
vocab_file = os.path.join(save_dir, artist_name + "_words_vocab.pkl")
sequence_length = 30
sequence_step = 1

# create word list
artist_word_list = create_wordlist(artist_form_lyrics_folder_path, artist_name)

'''
CREATE DICTIONARY
'''
# count the number of words
word_counts = collections.Counter(artist_word_list)
# Mapping from index to word : that's the vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

# size of the vocabulary
vocab_size = len(words)
print("vocab size: ", vocab_size)
with open(os.path.join(vocab_file), 'wb') as f:
    pickle.dump((words, vocab, vocabulary_inv), f)


'''
CREATE SEQUENCES
Now, we have to create the input data for our LSTM. We create two lists:

sequences: this list will contain the sequences of words used to train the model,
next_words: this list will contain the next words for each sequences of the sequences list.
'''
#create sequences
sequences = []
next_words = []
for i in range(0, len(artist_word_list) - sequence_length, sequence_step):
    sequences.append(artist_word_list[i: i + sequence_length])
    next_words.append(artist_word_list[i + sequence_length])

print('nb sequences:', len(sequences))

