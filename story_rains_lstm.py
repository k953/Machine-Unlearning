# -*- coding: utf-8 -*-
"""Story_rains_LSTM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fbSg_nAvDNqQK9E7IuZXskyzMBluRooz
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import regex as re

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

import os
data_directory = "/content/gdrive/My Drive/Short_Story"

filenames = []

for filename in os.listdir(data_directory):
  filenames.append(filename)

filenames[1]

path =  os.path.join(data_directory,filenames[1])
path

def file_to_sentence_list(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    # Splitting the text into sentences using
    # delimiters like '.', '?', and '!'
    sentences = [sentence.strip() for sentence in re.split(
        r'(?<=[.!?])\s+', text) if sentence.strip()]

    return sentences

file_path = os.path.join(data_directory,filename)
text_data = file_to_sentence_list(file_path)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1
print(total_words)
# Create input sequences
input_sequences = []
for line in text_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    #print(token_list)
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        #8-print(n_gram_sequence)
        input_sequences.append(n_gram_sequence)

# Pad sequences and split into predictors and label
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(
    input_sequences, maxlen=max_sequence_len, padding='pre'))

# print(input_sequences)
X, y = input_sequences[:, :-1], input_sequences[:, -1]

# Convert target data to one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print(total_words)

model = Sequential()
model.add(Embedding(total_words, 10,
                    input_length=max_sequence_len-1))
model.add(LSTM(128))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=75, verbose=1)



#The dinner dishes manipulated like magic tricks, and in the study a click
seed_text = "The dinner dishes"
next_words = 5

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = model.predict(token_list)
    predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
    seed_text += " " + predicted_word

print("Next predicted words:", seed_text)

model.save('story-rain-model.h5')

for filename in os.listdir(data_directory):
  print(filename)

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

import os
data_directory = "/content/gdrive/My Drive/Short_Story"

filenames = []

for filename in os.listdir(data_directory):
  filenames.append(filename)

filename = filenames[1]

filename

def file_to_sentence_list(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    # Splitting the text into sentences using
    # delimiters like '.', '?', and '!'
    sentences = [sentence.strip() for sentence in re.split(
        r'(?<=[.!?])\s+', text) if sentence.strip()]

    return sentences

file_path = os.path.join(data_directory,filename)
text_data = file_to_sentence_list(file_path)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1
print(total_words)
# Create input sequences
input_sequences = []
for line in text_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    #print(token_list)
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        #print(n_gram_sequence)
        input_sequences.append(n_gram_sequence)

# Pad sequences and split into predictors and label
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(
    input_sequences, maxlen=max_sequence_len, padding='pre'))

# print(input_sequences)
X, y = input_sequences[:, :-1], input_sequences[:, -1]

# print("_______ ")
# print(X)

# Convert target data to one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

from keras import models
model_new = models.load_model('/content/story-rain-model.h5')

model_new.fit(X, y, epochs=20, verbose=1)

#The dinner dishes a great spoors matted low, and in the study a click.
seed_text = "The dinner dishes"
next_words = 5

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = model.predict(token_list)
    predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
    seed_text += " " + predicted_word

print("Next predicted words:", seed_text)