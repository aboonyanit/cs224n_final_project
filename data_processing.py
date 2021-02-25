import numpy as np 
import pandas as pd 
import os
import csv

# This class does some data pre-processing (getting rid of duplicates, removing three genres with small
# genre counts, merging 2 CSV files, one-hot encoding the genres) to generate lyric-genre-data.csv.
# This class also generates lyrics.txt which tokenizes punctuation and writes lyrics to a text file 
# where each line in text file is lyrics for one song in order to create GloVe vectors.
# Parts of generating lyric-genre-data.csv code based off: https://www.kaggle.com/nkode611/lyricsgenreclassifier-datapreprocessing

df_artists = pd.read_csv('../cs224n_dataset/artists-data.csv')
df_artists_link_genre = df_artists[['Link', 'Genre']]
print('Number of artists with duplicate genres: ', df_artists_link_genre.duplicated(subset = 'Link', keep = 'first').value_counts()[True]) 

df_lyrics = pd.read_csv('../cs224n_dataset/lyrics-data.csv')
df_lyrics_en = df_lyrics.drop(df_lyrics[df_lyrics['Idiom'] !='ENGLISH'].index) #Get rid of non-English lyrics
# Drop duplicates in the field 'SLink'
df_lyrics_en.drop_duplicates(subset='SLink', keep='first', inplace=True, ignore_index=False)
df_lyrics_nd = df_lyrics_en.drop(['SName', 'SLink', 'Idiom'], axis=1)

# Discard all duplicate rows: 
df_lyrics_nd.drop_duplicates(inplace=True)
# Merge the two datasets to get a dataset with lyric and genre
df_merged = pd.merge(df_lyrics_nd, df_artists_link_genre, how='inner', left_on='ALink', right_on='Link')
df_lyric_genre = df_merged.drop(['ALink','Link'], axis=1)
print("Genre labels", df_lyric_genre.Genre.value_counts()) #Keep only pop, rock, and hip hop because other genres have very small counts
df_lyric_genre = df_lyric_genre.drop(df_lyric_genre[ (df_lyric_genre['Genre'] == 'Sertanejo') | (df_lyric_genre['Genre'] == 'Samba') | (df_lyric_genre['Genre'] == 'Funk Carioca')].index)
df_lyric_genre.drop_duplicates(inplace=True)

# Ordinal encode the genres b/c output column
df_lyric_genre.Genre = df_lyric_genre.Genre.replace({'Pop': 1, 'Rock': 2, 'Hip Hop': 3})
df_lyric_genre.to_csv('../cs224n_dataset/lyric-genre-data.csv', index = False)

f = open("lyrics.txt", "w")
f1 = open("../cs224n_dataset/lyric-genre-data-punctuation-separated.csv", "w")
filewriter = csv.writer(f1)
punctuation_chars = [",", ".", "\"", "?", "!"]
with open('../cs224n_dataset/lyric-genre-data.csv', 'r') as read_obj:
    # Write lyrics to a text file where each line in text file is lyrics for one song in order to create GloVe vectors
    csv_dict_reader = csv.DictReader(read_obj)
    for row in csv_dict_reader:
        lyrics = row['Lyric'].lower()
        punctuation_indices = []
        for punctuation in punctuation_chars:
            # Get indicies of punctuation in each song
            [punctuation_indices.append(i) for i, ltr in enumerate(lyrics) if ltr == punctuation]
        punctuation_indices.sort()
        for i, punctuation_ind in enumerate(punctuation_indices):
            # Add a space between the punctuation so that each piece of punctuation is considered its own word
            # i.e. the word "night." becomes tokenized as "night" and "."
            lyrics = lyrics[0: punctuation_ind + i] + " " + lyrics[punctuation_ind + i: ] 
        filewriter.writerow([lyrics, row['Genre']])
        print(lyrics, file=f)   
f.close()
f1.close()