from collections import Counter

import pandas as pd 
import csv
import matplotlib.pyplot as plt

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
# print(rock_words)
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
# ax = fig.add_axes([0,0,1,1])
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
# ax = fig.add_axes([0,0,1,1])
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
# ax = fig.add_axes([0,0,1,1])
print(hiphop_words)
print(hiphop_counts)
print(len(hiphop_words))
plt.bar(hiphop_words[0: 25], hiphop_counts[0: 25])
plt.show()
