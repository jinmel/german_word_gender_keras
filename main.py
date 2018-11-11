# ========================Load data=========================
import numpy as np
import pandas as pd
import string
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model

from data import read_dictionary, build_char_dict, prepare_data


# Each line is formatted as: "word {gender}"
word_list = read_dictionary('./wiktionary_nouns_with_gender.txt')
char2idx, idx2char = build_char_dict()

vocab_size = len(char2idx)

embedding_weights = []
embedding_weights.append(np.zeros(vocab_size))

for char, idx in char2idx.items():
    onehot = np.zeros(vocab_size)
    onehot[idx-1] = 1
    embedding_weights.append(onehot)

embedding_weights = np.array(embedding_weights)
embedding_size = embedding_weights.shape[1]

# input size is larger than max length of the word
input_size = len(max(word_list, key=lambda x: len(x[0]))[0]) + 5
input_size = 256
print('Input size:', input_size)
dropout = 0.5
num_classes = 3

layers = {}

layers['inputs'] = Input(shape=(input_size,), name='input', dtype='int32')
layers['embedding'] = Embedding(vocab_size + 1,
                                embedding_size,
                                input_length=input_size)(layers['inputs'])

layers['conv_1'] = Conv1D(32, 3)(layers['embedding'])
layers['conv_1_relu'] = Activation('relu')(layers['conv_1'])
layers['conv_1_max_pool'] = MaxPooling1D(pool_size=2)(layers['conv_1_relu'])

layers['conv_2'] = Conv1D(32, 3)(layers['conv_1_max_pool'])
layers['conv_2_relu'] = Activation('relu')(layers['conv_2'])
# layers['conv_2_max_pool'] = MaxPooling1D(pool_size=2)(layers['conv_2_relu'])

# layers['conv_3'] = Conv1D(32, 3)(layers['conv_2_max_pool'])
# layers['conv_3_relu'] = Activation('relu')(layers['conv_3'])
# layers['conv_4'] = Conv1D(32, 3)(layers['conv_3_relu'])
# layers['conv_4_relu'] = Activation('relu')(layers['conv_4'])
# layers['conv_5'] = Conv1D(32, 3)(layers['conv_4_relu'])
# layers['conv_5_relu'] = Activation('relu')(layers['conv_5'])

# layers['conv_6'] = Conv1D(256, 7)(layers['conv_5_relu'])
# layers['conv_6_relu'] = Activation('relu')(layers['conv_6'])
# layers['conv_6_max_pool'] = MaxPooling1D(pool_size=3)(layers['conv_6_relu'])

layers['flatten'] = Flatten()(layers['conv_2_relu'])

layers['fc_1'] = Dense(256, activation='relu')(layers['flatten'])
layers['fc_1_dropout'] = Dropout(dropout)(layers['fc_1'])

layers['fc_2'] = Dense(64, activation='relu')(layers['fc_1_dropout'])
layers['fc_2_dropout'] = Dropout(dropout)(layers['fc_2'])

layers['prediction'] = Dense(num_classes,
                             activation='softmax')(layers['fc_2_dropout'])

model = Model(inputs=layers['inputs'], outputs=layers['prediction'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# prepare training and testing dataset

# shuffle
random.shuffle(word_list)

inputs, labels = prepare_data(word_list, input_size, char2idx)
ratio = 0.9

train_count = int(len(inputs) * ratio)
test_count = len(inputs) - train_count

print('Train words: {} Test words: {}'.format(train_count, test_count))
print('Total: {}'.format(train_count + test_count))

train_inputs, test_inputs = inputs[:train_count], inputs[train_count:]
train_labels, test_labels = labels[:train_count], labels[train_count:]

# for input_data, label in zip(train_inputs, train_labels):
#     print(input_data, label)
#     input()

model.fit(train_inputs, train_labels,
          validation_data=(test_inputs, test_labels),
          batch_size=64,
          epochs=10,
          verbose=1)
