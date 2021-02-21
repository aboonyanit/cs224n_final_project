from csv import DictReader

f = open("lyrics.txt", "w")
with open('../cs224n_dataset/lyric-genre-data.csv', 'r') as read_obj:
    # Write lyrics to a text file where each line in text file is lyrics for one song
    csv_dict_reader = DictReader(read_obj)
    for row in csv_dict_reader:
        lyrics = row['Lyric']
        print(lyrics, file=f)   
f.close()

#should we increase number of iters to > 15
#should the and The be the same - to lower everything? - how about punctuation?