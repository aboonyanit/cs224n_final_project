import torch
import torch.nn as nn
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import csv
import pandas as pd
import statistics
from typing import List

# This file creates and runs a Logistic Regression model. It contains methods necessary to generate
# embeddings and add necessary padding. Small amounts of the code was based off of Assignment 4.


def generate_embeddings(file_path):
    """ Generates an embeddings vector which contains the pre-trained GloVe vectors we generated.
    It also generates a list of the vocabulary.

    @param file_path (str): File path containing the GloVe vectors

    @returns vocab: list of vocab (number of vocab words)
    @returns embeddings: matrix of embeddings (vocabulary size, embedding dimension)
    """
    embeddings_dict = {} # each key is word and value is vector of floats - not currently using it - can probably delete later
    embeddings = []
    vocab = []
    with open(file_path, "r") as f:
        for line in f:
            split_line = line.split(" ")
            vector = [float(i) for i in split_line[1: ]]
            embeddings_dict[split_line[0]] = vector
            embeddings.append(vector)
            vocab.append(split_line[0])
            #Add unk token to dictionary
    vocab.append('<pad>')
    embeddings.append([0] * 100) # Currently made the pad token 100 0's but can change that
    return vocab, embeddings
    
def to_input_tensor(self, lyrics_list: List[List[str]], device: torch.device) -> torch.Tensor:
    """ Convert list of sentences (words) into tensor with necessary padding for 
    shorter sentences.

    @param lyrics_list (List[List[str]]): list of lyrics (words)
    @param device: device on which to load the tesnor, i.e. CPU or GPU

    @returns lyrics_var: tensor of (longest_lyric_len, batch_size)
    """
    lyrics_var = []
    longest_lyric_len = len(max(lyrics_list, key=len))
    for lyrics in lyrics_list:
        num_pads_to_add = longest_lyric_len - len(lyrics)
        lyrics += ["<pad>"] * num_pads_to_add
        lyrics_indicies = []
        for word in lyrics:
            if word not in self.word2indicies.keys():
                lyrics_indicies.append(self.word2indicies['<unk>'])
            else:
                lyrics_indicies.append(self.word2indicies[word])
        lyrics_var.append(lyrics_indicies)
    lyrics_var = torch.tensor(lyrics_var, dtype=torch.long, device=device)
    return torch.t(lyrics_var)

class LogisticRegression(nn.Module):
    """ Simple Logistic Regression Model
    """
    def __init__(self, vocab_size, embedding_dim, embeddings, vocab, n_classes=3):
        """ Init LogisticRegression Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(LogisticRegression, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).from_pretrained(torch.FloatTensor(embeddings))
        self.word2indicies = {word: ind for ind, word in enumerate(vocab)}

    def forward(self, lyrics: List[List[str]]) -> torch.Tensor:
        # Convert list of lists into tensors
        lyrics_padded = to_input_tensor(self, lyrics, device=self.device) 

if __name__ == '__main__':
    vocab, embeddings = generate_embeddings('vectors.txt')
    model = LogisticRegression(len(vocab), len(embeddings[0]), embeddings, vocab)

