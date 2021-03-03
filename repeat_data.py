import csv

scratch_file = open("../cs224n_dataset/scratch-file.csv", "w")
filewriter = csv.writer(scratch_file)
filewriter.writerow(['Lyric', 'Hip Hop', 'Pop', 'Rock']) #??

with open("../cs224n_dataset/test-data.csv", "r") as read_obj:
    csv_dict_reader = csv.DictReader(read_obj)
    for row in csv_dict_reader:
        if row['Hip Hop'] == "1":
            # duplicate twice
            filewriter.writerow([row['Lyric'], row['Hip Hop'], row['Pop'], row['Rock']])
            filewriter.writerow([row['Lyric'], row['Hip Hop'], row['Pop'], row['Rock']])
        if row['Pop'] == "1":
            filewriter.writerow([row['Lyric'], row['Hip Hop'], row['Pop'], row['Rock']])
read_obj.close()

train_dataset = open("../cs224n_dataset/test-data.csv", "a")
filewriter = csv.writer(train_dataset)

with open("../cs224n_dataset/scratch-file.csv", "r") as read_obj:
    csv_dict_reader = csv.DictReader(read_obj)
    for row in csv_dict_reader:
        filewriter.writerow([row['Lyric'], row['Hip Hop'], row['Pop'], row['Rock']])