import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

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
print("Genre labels", df_lyric_genre.Genre.value_counts()) #Keep only pop, rock, and hip hop
df_lyric_genre = df_lyric_genre.drop(df_lyric_genre[ (df_lyric_genre['Genre'] == 'Sertanejo') | (df_lyric_genre['Genre'] == 'Samba') | (df_lyric_genre['Genre'] == 'Funk Carioca')].index)
df_lyric_genre.drop_duplicates(inplace=True)

# One hot encode the genres (three columns - "pop", "rock", "hip hop")
df_lyric_genre = pd.concat([df_lyric_genre, pd.get_dummies(df_lyric_genre['Genre'])],axis=1)
# Drop the original "Genre" column 
df_lyric_genre.drop(['Genre'],axis=1, inplace=True)

df_lyric_genre.to_csv('../cs224n_dataset/lyric-genre-data.csv', index = False)