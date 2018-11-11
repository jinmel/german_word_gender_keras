import numpy as np
import string
from keras.utils import to_categorical

UNK = 'UNK'

def pad_zero(sequence, max_length):
    n_pad = max_length - sequence.shape[0]
    return np.concatenate(sequence, np.zeros(n_pad))

def read_dictionary(file_path):
    word_list = []

    with open('./wiktionary_nouns_with_gender.txt') as german_dict_f:
        lines = german_dict_f.readlines()

        for line in lines:
            word, gender = line.split(' ')
            word = word.lower()
            # gender is {m} or {f} or {n}
            gender = gender[1]
            word_list.append((word, gender))

    return word_list

def build_char_dict():
    alphabets = string.ascii_letters + 'äöüß'

    char2idx = {}

    for i, char in enumerate(alphabets, 1):
        char2idx[char] = i

    char2idx[UNK] = len(char2idx) + 1
    idx2char = {v: k for k,v in char2idx.items()}

    return char2idx, idx2char

def prepare_data(word_gender_pairs, input_size, char2idx):
    inputs = []
    labels = []
    gender2idx= {'m': 0, 'f': 1, 'n': 2}
    for word, gender in word_gender_pairs:
        seq = []
        for ch in word:
            if ch in char2idx:
                seq.append(char2idx[ch])
            else:
                seq.append(char2idx[UNK])

        seq = np.array(seq)
        seq = np.pad(seq, (0, input_size - seq.shape[0]), 'constant')
        assert seq.shape[0] == input_size
        inputs.append(seq)
        labels.append(gender2idx[gender])

    return np.array(inputs), to_categorical(labels)
