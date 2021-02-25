import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../cs224n_dataset/lyric-genre-data-punctuation-separated.csv')

train, test = train_test_split(df, test_size=0.2, random_state=42)
train.to_csv('../cs224n_dataset/train-data.csv', index = False)
test.to_csv('../cs224n_dataset/test-data.csv', index = False)

df = pd.read_csv('../cs224n_dataset/test-data.csv')
validation, test = train_test_split(df, test_size=0.5, random_state=42)
validation.to_csv('../cs224n_dataset/validation-data.csv', index = False)
test.to_csv('../cs224n_dataset/test-data.csv', index = False)

