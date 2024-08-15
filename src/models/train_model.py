import numpy as np 
import pandas as pd 
import random
import os
import tensorflow as tf
from tqdm import tqdm
import re
import pathlib

import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Embedding, LSTM, Concatenate, Dropout,
                                     Input, Dense, Bidirectional, Layer, TimeDistributed)

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, LayerNormalization, MultiHeadAttention, Add, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def tokenizer(df, col):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df[col].values)
    vocab_size = len(tokenizer.word_index) + 1
    encoded_sequences = tokenizer.texts_to_sequences(df[col].values)
    return tokenizer, vocab_size, encoded_sequences

def pad_sequences_for_languages(dyu_sequences, fr_sequences, dyu_lengths, fr_lengths):
    max_dyu_len = dyu_lengths.max()
    max_fr_len = fr_lengths.max()
    
    dyu_padded = pad_sequences(dyu_sequences, maxlen=max_dyu_len, padding='post')
    fr_padded = pad_sequences(fr_sequences, maxlen=max_fr_len, padding='post')
    
    return dyu_padded, fr_padded, max_dyu_len, max_fr_len


def load_glove_embeddings(glove_file, embedding_dim=300):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding_vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding_vector
    return embeddings_index


def create_embedding_matrix(tokenizer, embeddings_index, embedding_dim=300):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def load_and_create_embedding_matrices(glove_file, dyu_tokenizer, fr_tokenizer, embedding_dim=300):
    embeddings_index = load_glove_embeddings(glove_file, embedding_dim)
    dyu_embedding_matrix = create_embedding_matrix(dyu_tokenizer, embeddings_index, embedding_dim)
    fr_embedding_matrix = create_embedding_matrix(fr_tokenizer, embeddings_index, embedding_dim)
    return dyu_embedding_matrix, fr_embedding_matrix

if __name__ == "__main__":
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    glove_file = home_dir.as_posix() + '/models/glove.6B.300d.txt'

    train_path = home_dir.as_posix() + '/data/processed/train.csv'
    validate_path = home_dir.as_posix() + '/data/processed/validate.csv'
    test_path = home_dir.as_posix() + '/data/processed/test.csv'

    train = load_data(train_path)
    validate = load_data(validate_path)
    test = load_data(test_path)

    dyu_tokenizer, dyu_vocab_size, dyu_sequences = tokenizer(train, 'dyu')
    fr_tokenizer, fr_vocab_size, fr_sequences = tokenizer(train, 'fr')

    dyu_lengths = np.array([len(seq) for seq in dyu_sequences])
    fr_lengths = np.array([len(seq) for seq in fr_sequences])

    dyu_padded, fr_padded, max_dyu_len, max_fr_len = pad_sequences_for_languages(dyu_sequences, fr_sequences, dyu_lengths, fr_lengths)

    embedding_dim = 300

    
    # Assuming dyu_tokenizer and fr_tokenizer are already defined
    dyu_embedding_matrix, fr_embedding_matrix = load_and_create_embedding_matrices(glove_file, dyu_tokenizer, 
                                                                                   fr_tokenizer, embedding_dim)
    
    print('code running successfully')
    