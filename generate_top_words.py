from collections import Counter

import pandas as pd 
import csv
import matplotlib.pyplot as plt

num_hip_hop = 0
sum_hip_hop = 0
num_pop = 0
sum_pop = 0
num_rock = 0
sum_rock = 0
num_lyrics = 0
sum_lyrics = 0
with open('../cs224n_dataset/lyric-genre-data.csv', 'r') as read_obj:
    # Get average song length of different genres
    csv_dict_reader = csv.DictReader(read_obj)
    for row in csv_dict_reader:
        lyric_len = len(row["Lyric"].split(' '))
        sum_lyrics += lyric_len
        num_lyrics += 1
        if row["Hip Hop"] == '1':
            num_hip_hop += 1
            sum_hip_hop += lyric_len
        elif row["Pop"] == '1':
            num_pop += 1
            sum_pop += lyric_len
        elif row["Rock"] == '1':
            num_rock += 1
            sum_rock += lyric_len
print("Avg song length", sum_lyrics / num_lyrics)
print("Avg hip hop length", sum_hip_hop / num_hip_hop)
print("Avg pop length", sum_pop / num_pop)
print("Avg rock length", sum_rock / num_rock)

rock_words = []
pop_words = []
hiphop_words = []

with open('../cs224n_dataset/lyric-genre-data-punctuation-separated.csv', 'r') as read_obj:
    csv_dict_reader = csv.DictReader(read_obj)
    for row in csv_dict_reader:
        lyrics = row['Lyric'].split(" ")
        if row['Genre'] == '1':
            pop_words += lyrics
        elif row['Genre'] == '2':
            rock_words += lyrics
        else:
            hiphop_words += lyrics
rock_words_dict = Counter(rock_words)
pop_words_dict = Counter(pop_words)
hiphop_words_dict = Counter(hiphop_words)

unique_top_rock = []
unique_top_pop = []
unique_top_hiphop = []

# print(rock_words_dict.most_common(100))
# print(pop_words_dict.most_common(100))
# print(hiphop_words_dict.most_common(100))
hiphop_words = []
hiphop_counts = []
top_words = [a_tuple[0] for a_tuple in pop_words_dict.most_common(300)] + [a_tuple[0] for a_tuple in hiphop_words_dict.most_common(300)]
for word_pair in rock_words_dict.most_common(300):
    if word_pair[0] not in top_words:
        hiphop_words.append(word_pair[0])
        hiphop_counts.append(word_pair[1])
fig = plt.figure()
print(hiphop_words)
print(hiphop_counts)
print(len(hiphop_words))
plt.bar(hiphop_words, hiphop_counts)
plt.show()

print("pop")
pop_words = []
pop_counts = []
top_words = [a_tuple[0] for a_tuple in rock_words_dict.most_common(300)] + [a_tuple[0] for a_tuple in hiphop_words_dict.most_common(300)]
for word_pair in pop_words_dict.most_common(300):
    if word_pair[0] not in top_words:
        pop_words.append(word_pair[0])
        pop_counts.append(word_pair[1])

fig = plt.figure()
print(pop_words)
print(pop_counts)
print(len(pop_words))
plt.bar(pop_words, pop_counts)
plt.show()

print("hip hop")
hiphop_words = []
hiphop_counts = []
top_words = [a_tuple[0] for a_tuple in pop_words_dict.most_common(200)] + [a_tuple[0] for a_tuple in rock_words_dict.most_common(200)]
for word_pair in hiphop_words_dict.most_common(300):
    if word_pair[0] not in top_words and word_pair[0] != "[verse" and word_pair[0] != "[chorus]":
        hiphop_words.append(word_pair[0])
        hiphop_counts.append(word_pair[1])

fig = plt.figure()
print(hiphop_words)
print(hiphop_counts)
print(len(hiphop_words))
plt.bar(hiphop_words[0: 25], hiphop_counts[0: 25])
plt.show()
