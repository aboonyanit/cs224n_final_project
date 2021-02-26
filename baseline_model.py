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
        lyrics = list(lyrics) + (["<pad>"] * num_pads_to_add)
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
        self.linear = nn.Linear(4571, n_classes) #lenght of a padded sentence? - 
        #2936 was the longest lyric length for test, 4571 is longest for training data
        self.device = torch.device("cpu")

    def forward(self, lyrics:torch.LongTensor ) -> torch.Tensor:#List[List[str]]
        # Convert list of lists into tensors
        # lyrics_padded = to_input_tensor(self, lyrics, device=self.device) 
        print("forward")
        print(lyrics.shape)
        #(number of samples in batch/x_train, length of padded sentence) * (length of padded sentence, 3)
        # dimensions of lyrics * dimensions of linear layer
        #nn.Linear() expects a tensor of size (N_examples, n_features)
        return self.linear(torch.transpose(lyrics, 0, 1))
        return self.linear(lyrics)

if __name__ == '__main__':
    vocab, embeddings = generate_embeddings('vectors.txt')
    #create model
    device = torch.device('cpu')
    model = LogisticRegression(len(vocab), len(embeddings[0]), embeddings, vocab)
    model = model.to(device)
    # get data
    testCSV = pd.read_csv("../cs224n_dataset/test-data.csv")
    trainCSV = pd.read_csv("../cs224n_dataset/train-data.csv")

    x_test_csv = [i.split(' ') for i in testCSV["text"].values]
    x_train_csv = [i.split(' ') for i in trainCSV["text"].values]
    y_test_csv = [float(i) for i in testCSV["label"]]
    y_train_csv = [float(i) for i in trainCSV["label"]]

    x_test = torch.LongTensor(to_input_tensor(model, lyrics_list = x_test_csv, device=device))
    x_train = torch.LongTensor(to_input_tensor(model, lyrics_list =x_train_csv, device=device))    
    y_test = torch.Tensor(y_test_csv)
    y_train =  torch.Tensor(y_train_csv)
    
    #hyperparameters
    learning_rate = 0.5
    epochs = 20

    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #train model
    for epoch in range(epochs):
        print("epoch")
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backwards()
        optimizer.step()

    #test model 
    eval_y_pred = model(x_test)



