from __future__ import print_function
import re
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy
import numpy as np
import os
import collections
import pickle


def load_list(root_path, author_name):
    artist_name_for_document_name = re.sub("[^0-9a-zA-Z]+", "_", author_name)
    dictionary_file_path = root_path + artist_name_for_document_name + "/"

    with open(dictionary_file_path + artist_name_for_document_name + '_word_list.pkl', 'rb') as f:
        return pickle.load(f)


def load_dictionary(root_path, author_name):
    artist_name_for_document_name = re.sub("[^0-9a-zA-Z]+", "_", author_name)
    dictionary_file_path = root_path + artist_name_for_document_name + "/"

    with open(dictionary_file_path + artist_name_for_document_name + '.pkl', 'rb') as f:
        return pickle.load(f)


def create_wordlist(artist_form_lyrics_folder_path, artist_name):
    loaded_list = load_list(artist_form_lyrics_folder_path, artist_name)
    return loaded_list


def create_word_list_and_word_dictionary(artist_word_list):
    # count the number of words
    word_counts = collections.Counter(artist_word_list)
    # Mapping from index to word : that's the vocabulary
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocab = {x: i for i, x in enumerate(vocabulary_inv)}
    words = [x[0] for x in word_counts.most_common()]
    return vocabulary_inv, vocab, words


def save_vocabulary_and_list_of_words(vocab_file, words, vocab, vocabulary_inv):
    with open(os.path.join(vocab_file), 'wb') as f:
        pickle.dump((words, vocab, vocabulary_inv), f)


def create_sequences_of_words_and_next_word_for_each_sequence(artist_word_list, sequence_length, sequence_step):
    '''
    sequences: this list will contain the sequences of words used to train the model,
    next_words: this list will contain the next words for each sequences of the sequences list.
    '''
    sequences = []
    next_words = []
    for i in range(0, len(artist_word_list) - sequence_length, sequence_step):
        sequences.append(artist_word_list[i: i + sequence_length])
        next_words.append(artist_word_list[i + sequence_length])
    return sequences, next_words


def build_LSTM_input_matrixes(sequences, sequence_length, vocab_size, vocab):
    X = np.zeros((len(sequences), sequence_length, vocab_size), dtype=np.bool)
    y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, word in enumerate(sentence):
            X[i, t, vocab[word]] = 1
        y[i, vocab[next_words[i]]] = 1
    return X, y


def bidirectional_lstm_model(seq_length, vocab_size, add_attention):
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"), input_shape=(seq_length, vocab_size)))
    if add_attention:
        model.add(Activation('softmax'))
    model.add(Dropout(0.4))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


'''
MAIN
'''
artist_form_lyrics_folder_path = "../data/rap_lyrics_links/"
artist_name = "Eminem"
save_dir = './words_vocabulary_files_and_models/'
vocab_file = os.path.join(save_dir, artist_name + "_words_vocab.pkl")
sequence_length = 6  # from 30 to 8
sequence_step = 3  # from 1 to 3

rnn_size = 256  # size of RNN
batch_size = 32  # minibatch size
num_epochs = 25  # number of epochs from 50 do 25
learning_rate = 0.001  # learning rate, form 0.001 (best), to 0.01, to 0.0001
add_attention = True

artist_word_list = create_wordlist(artist_form_lyrics_folder_path, artist_name)

vocabulary_inv, vocab, words = create_word_list_and_word_dictionary(artist_word_list)
# size of the vocabulary
vocab_size = len(words)
print("vocab size: ", vocab_size)
save_vocabulary_and_list_of_words(vocab_file, words, vocab, vocabulary_inv)

sequences, next_words = create_sequences_of_words_and_next_word_for_each_sequence(artist_word_list, sequence_length,
                                                                                  sequence_step)
print('nb sequences:', len(sequences))

# X, y = build_LSTM_input_matrixes(sequences, sequence_length, vocab_size, vocab)
#
# bidirectional_lstm = bidirectional_lstm_model(sequence_length, vocab_size, add_attention)
# from keras.models import load_model
# bidirectional_lstm = load_model(save_dir + 'my_model_gen_sentences_lstm.final.hdf5')
# bidirectional_lstm.summary()
#
# traininga
# callbacks = [EarlyStopping(patience=3, monitor='val_loss'),
#              ModelCheckpoint(filepath=save_dir + 'my_model_gen_sentences_lstm.{epoch:02d}-{val_loss:.2f}.hdf5',
#                              monitor='val_loss', save_best_only=True, verbose=0, mode='auto', period=1)]
# history = bidirectional_lstm.fit(X, y,
#                                  batch_size=batch_size,
#                                  shuffle=True,
#                                  epochs=num_epochs,
#                                  callbacks=callbacks,
#                                  validation_split=0.1)  # validation split sa 0.01 na 0.1 povecan


# # save the model
# bidirectional_lstm.save(save_dir + 'my_model_gen_sentences_lstm.final.hdf5')

from keras.models import load_model

bidirectional_lstm = load_model(save_dir + 'my_model_gen_sentences_lstm.final.hdf5')

# initiate sentences
seed_sentences = "<verse_start> DHAH0 WEY1 SHIY1 MUW1VZ , SHIY1Z LAY1K AH0 BEH1LYAH0DNAH0SER0 <end_line>"
generated = ''
sentence = []
for i in range(sequence_length):
    sentence.append("a")

seed = seed_sentences.split()

for i in range(sequence_length):
    sentence[sequence_length - i - 1] = seed[len(seed) - i - 1]

generated += ' '.join(sentence)
print('Generating text with the following seed: "' + ' '.join(sentence) + '"')

words_number = 150 # how much words to generate
artist_dictionary = load_dictionary(artist_form_lyrics_folder_path, artist_name)
# generate the text
for i in range(words_number):
    # create the vector
    x = np.zeros((1, sequence_length, vocab_size))
    for t, word in enumerate(sentence):
        x[0, t, vocab[word]] = 1.
    # print(x.shape)

    # calculate next word
    preds = bidirectional_lstm.predict(x, verbose=0)[0]
    next_index = sample(preds)
    next_word = vocabulary_inv[next_index]

    next_word_grapheme = artist_dictionary[next_word]
    # add the next word to the text
    generated += " " + next_word_grapheme
    # shift the sentence by one, and and the next word at its end
    sentence = sentence[1:] + [next_word]

print(generated)
