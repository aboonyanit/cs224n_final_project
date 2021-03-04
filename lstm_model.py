import torch
import torch.nn as nn
import spacy
import csv
import pandas as pd
import statistics
from typing import List
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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
            embeddings_dict[split_line[0]] = np.asarray(vector)
            embeddings.append(vector)
            vocab.append(split_line[0])
            #Add unk token to dictionary
    vocab.append('<pad>')
    embeddings.append([0] * 100) # Currently made the pad token 100 0's but can change that
    return vocab, embeddings
    
def to_input_tensor(self, lyrics_list: List[List[str]], max_len_padded_seq, device) -> torch.Tensor:
    """ Convert list of sentences (words) into tensor with necessary padding for 
    shorter sentences.
    @param lyrics_list (List[List[str]]): list of lyrics (words)
    @param device: device on which to load the tesnor, i.e. CPU or GPU
    @returns lyrics_var: tensor of (longest_lyric_len, batch_size)
    """
    lyrics_var = []
    # longest_lyric_len = len(max(lyrics_list, key=len))
    for lyrics in lyrics_list:
        num_pads_to_add = max_len_padded_seq - len(lyrics)
        lyrics_indicies = []
        for i, word in enumerate(lyrics):
            if word not in self.word2indicies.keys():
                lyrics_indicies.append(self.word2indicies['<unk>'])
            else:
                lyrics_indicies.append(self.word2indicies[word])
            if i == max_len_padded_seq - 1:
                break
        lyrics_indicies += ([self.word2indicies['<pad>']] * num_pads_to_add)
        lyrics_var.append(lyrics_indicies)
    lyrics_var = torch.FloatTensor(lyrics_var).to(device)
    return lyrics_var

def validation_metrics (model, valid_dl, epoch):
    # test model on val set
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    nb_classes = 3
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    index = 0
    for x, y, l in valid_dl:
        x = x.long()
        y = torch.argmax(y, 1)
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        pred = pred.cpu()
        y=y.cpu()
        for t, p in zip(y.view(-1), pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
            index += 1
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    print(confusion_matrix)
    print("Per class accuracy", confusion_matrix.diag() / confusion_matrix.sum(1))

    confusion_matrix_df = pd.DataFrame(confusion_matrix, index=['Hip Hop', 'Pop', 'Rock'], columns=['Hip Hop', 'Pop', 'Rock']).astype("float")
    if epoch == 49:
        sns.heatmap(confusion_matrix_df, annot=True, fmt='g')
        plt.show()
        plt.savefig("confusion_matrix_unbalanced.png") #does this work?

    return sum_loss/total, correct/total, sum_rmse/total

class LyricsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.to(device)
        self.y = Y.to(device)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], len(self.X[idx])

class LSTM_model(nn.Module):
    """ Simple LSTM
    """
    def __init__(self, vocab_size, embedding_dim, embeddings, vocab, hidden_dim, num_layers, n_classes=3):
        """ Init LSTM_model Model.
        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(LSTM_model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).from_pretrained(torch.FloatTensor(embeddings))
        self.word2indicies = {word: ind for ind, word in enumerate(vocab)}
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_classes) 
        self.dropout = nn.Dropout(0.2)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def forward(self, lyrics:torch.LongTensor, totalLen ) -> torch.Tensor:#List[List[str]]
        # Convert list of lists into tensors
        x = self.embedding(lyrics)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, totalLen, batch_first=True, enforce_sorted=False)
        out_pack, (hidden_state, cell_state) = self.lstm(x_pack)
        out = self.linear(hidden_state[-1]).to(device)
        return out

if __name__ == '__main__':
    print('initializing...')
    vocab, embeddings = generate_embeddings('vectors.txt')
    pad_token_index = len(vocab) - 1
    max_len_padded_seq = 500
    #create model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = LSTM_model(len(vocab), len(embeddings[0]), embeddings, vocab, num_layers=1, hidden_dim=50)
    model = model.to(device)
    # get data
    valCSV = pd.read_csv("../cs224n_dataset/validation-data-unbalanced.csv")
    trainCSV = pd.read_csv("../cs224n_dataset/train-data-unbalanced.csv")

    x_val_csv = [i.split(' ') for i in valCSV["Lyric"].values]
    x_train_csv = [i.split(' ') for i in trainCSV["Lyric"].values]
    y_train_csv = []
    y_val_csv = []
    for index, row in trainCSV.iterrows():
        y_train_csv.append([row["Hip Hop"], row["Pop"], row["Rock"]])
    for index, row in valCSV.iterrows():
        y_val_csv.append([row["Hip Hop"], row["Pop"], row["Rock"]])

    x_val = to_input_tensor(model, lyrics_list = x_val_csv, max_len_padded_seq=max_len_padded_seq, device=device).to(device)
    x_train = to_input_tensor(model, lyrics_list = x_train_csv, max_len_padded_seq=max_len_padded_seq, device=device).to(device)
    y_val = torch.Tensor(y_val_csv).to(device)
    y_train =  torch.Tensor(y_train_csv).to(device)

    train_dataset = LyricsDataset(x_train, y_train)
    val_dataset = LyricsDataset(x_val, y_val)
    batch_size=128
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)  

    #hyperparameters
    learning_rate = 0.001
    batch_size = 128    
    epochs = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs_arr = []
    train_losses = []

    #train model
    print("training on lr "+str(learning_rate)+" batch "+str(batch_size)+" ...")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    for i in range(epochs):
        epochs_arr.append(i)
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_loader:
            x = x.long()
            y = torch.argmax(y, 1)
            y = y.long()
            lengths_without_pad = []
            for elem in x.tolist():
                # Get actual lengths of lyrics without pad tokens
                if pad_token_index in elem:
                    lengths_without_pad.append(elem.index(pad_token_index))
                else:
                    lengths_without_pad.append(max_len_padded_seq)
            y_pred = model(x, torch.LongTensor(lengths_without_pad))
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]
        # print("train loss ", sum_loss/total)
        val_loss, val_acc, val_rmse = validation_metrics(model, val_loader, i)
        train_losses.append(sum_loss/total)
        print("epoch %.3f, train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (i, sum_loss/total, val_loss, val_acc, val_rmse))
        training_graph, ax = plt.subplots()
        ax.plot(epochs_arr, train_losses)
        pltName = "lr_"+str(learning_rate)+"_batch_"+str(batch_size)+"_train_loss_unbalanced.png"
        training_graph.savefig(pltName)
        plt.close(training_graph)

