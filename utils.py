import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(infile):
    '''
    Load & prepare fasta protein sequences.
    '''
    # Loading the protein fasta sequences; csv file containing sequences,
    # their labels, and seuence names
    data = pd.read_csv(infile, header=0, sep=',')

    # Get the set of all letters (single letter amino acid codes) in the data
    char_set = list(set("".join(data['sequence'])))
    vocab_size = len(char_set)
    # Creat a dictionary of aino acid letters and their index value in the list.
    # So that we have a numerical code for each letter. We start with 1 to save
    # 0 for padding.
    vocab = dict(zip(char_set, range(1, vocab_size+1)))

    # Embedding the characters using the dictioary built above. Basically,
    # replacing  each character with its numerical code in the dictionary.
    data_embed = pd.DataFrame([list(map(vocab.get, k)) for k in data['sequence']])

    # Replace nan with 0; equivalent of padding with 0
    data_embed[np.isnan(data_embed)] = 0

    # Rename columns
    data_embed.columns = ['S'+str(x+1) for x in range(data_embed.shape[1])]

    # Converting categorical labels to actual numbers For two categories of
    # labels, for example, we'll have classes [0, 1].
    labels = np.array(data['label'].astype('category').cat.codes)

    # Add labels column to dataframe
    data_embed = data_embed.assign(label=labels)

    # Convert to float values to int
    data_embed = data_embed.astype(int)

    # Split data into train & test dataset; using a constant random seed for consistency
    train_df, test_df = \
        train_test_split(data_embed, test_size=0.2, random_state=42)

    # Get the embedded train & test sequences as numpy array
    train_seqs = np.array(train_df.drop('label', axis=1))
    test_seqs = np.array(test_df.drop('label', axis=1))

    # Get labels as numy arrays
    train_lebels = np.array(train_df['label'])
    test_lebels = np.array(test_df['label'])

    return train_seqs, test_seqs, train_lebels, test_lebels

def next_batch(x_data, y_data, seqs_len, batch_size):
    '''
    Returns batches of x & y
    '''
    # Choose a random set of row indices with the size of batch_size
    idx = np.random.choice(np.arange(len(x_data)), size=batch_size, replace=False)
    # Return the subset (batch) of data using the randomly chosen row indices
    return x_data[idx, :], y_data[idx], np.array(seqs_len)[idx]
