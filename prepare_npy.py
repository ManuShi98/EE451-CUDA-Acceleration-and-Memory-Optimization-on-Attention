import bz2
from collections import Counter
import re
import nltk
import numpy as np
import random
from tqdm.notebook import tqdm

random.seed(8)

nltk.download("punkt")

train_file_all = bz2.BZ2File("./data/train.ft.txt.bz2").readlines()
test_file_all = bz2.BZ2File("./data/test.ft.txt.bz2").readlines()

print("Number of training reivews: " + str(len(train_file_all)))
print("Number of test reviews: " + str(len(test_file_all)))

num_train = (
    80000  # we're training on the 80000 randomly selected reviews in the dataset
)
num_test = 20000  # Using 20000 reviews from test set

train_file = [x.decode("utf-8") for x in random.sample(train_file_all, num_train)]
test_file = [x.decode("utf-8") for x in random.sample(test_file_all, num_test)]

print(train_file[0])

del test_file_all, train_file_all
# Extracting labels from sentences

train_labels = [0 if x.split(" ")[0] == "__label__1" else 1 for x in train_file]
train_sentences = [x.split(" ", 1)[1][:-1].lower() for x in train_file]


test_labels = [0 if x.split(" ")[0] == "__label__1" else 1 for x in test_file]
test_sentences = [x.split(" ", 1)[1][:-1].lower() for x in test_file]

# Some simple cleaning of data

for i in range(len(train_sentences)):
    train_sentences[i] = re.sub("\d", "0", train_sentences[i])

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub("\d", "0", test_sentences[i])

# Modify URLs to <url>

for i in range(len(train_sentences)):
    if (
        "www." in train_sentences[i]
        or "http:" in train_sentences[i]
        or "https:" in train_sentences[i]
        or ".com" in train_sentences[i]
    ):
        train_sentences[i] = re.sub(
            r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i]
        )

for i in range(len(test_sentences)):
    if (
        "www." in test_sentences[i]
        or "http:" in test_sentences[i]
        or "https:" in test_sentences[i]
        or ".com" in test_sentences[i]
    ):
        test_sentences[i] = re.sub(
            r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i]
        )

print(train_labels[0], train_sentences[0])


# Dictionary that will map a word to the number of times it appeared in all the training sentences
words = Counter()

for i, sentence in enumerate(tqdm(train_sentences)):
    # The sentences will be stored as a list of words/tokens
    train_sentences[i] = []
    for word in nltk.word_tokenize(sentence):  # Tokenizing the words
        words.update([word.lower()])  # Converting all the words to lower case
        train_sentences[i].append(word)
        
        
print(train_sentences[0])

print(words["stuning"])

min_occurrence_threshold = 1
# Removing the words that only appear once
words = {k: v for k, v in words.items() if v > min_occurrence_threshold}
# Sorting the words according to the number of appearances, with the most common word being first
words = sorted(words, key=words.get, reverse=True)
# Adding padding and unknown to our vocabulary so that they will be assigned an index
words = ["_PAD", "_UNK"] + words
# Dictionaries to store the word to index mappings and vice versa
word2idx = {o: i for i, o in enumerate(words)}
idx2word = {i: o for i, o in enumerate(words)}

print(idx2word[0], idx2word[1])

import pickle

# Saving the word2idx, idx2word dictionary to a file
with open("./data/word2idx.pickle", "wb") as f:
    pickle.dump(word2idx, f)
    
with open("./data/idx2word.pickle", "wb") as f:
    pickle.dump(idx2word, f)

for i, sentence in enumerate(tqdm(train_sentences)):
    # Looking up the mapping dictionary and assigning the index to the respective words
    train_sentences[i] = [
        word2idx[word] if word in word2idx else 1 for word in sentence
    ]
    
for i, sentence in enumerate(tqdm(test_sentences)):
    # For test sentences, we have to tokenize the sentences as well
    test_sentences[i] = [
        word2idx[word.lower()] if word.lower() in word2idx else 1
        for word in nltk.word_tokenize(sentence)
    ]
    
train_sentences[10]

# Defining a function that either shortens sentences or pads sentences with 0 to a fixed length/ Pad from right
def pad_input(sentences, seq_len):
    features = np.zeros(
        (len(sentences), seq_len), dtype=int
    )  # instances * features (seq_len)
    for idx, review in enumerate(tqdm(sentences)):
        if len(review) != 0:
            features[idx, -len(review) :] = np.array(review)[:seq_len]
    return features

seq_len = 256  # The length that the sentences will be padded/shortened to

train_sentences = pad_input(train_sentences, seq_len)
test_sentences = pad_input(test_sentences, seq_len)

# Converting our labels into numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print(train_sentences[10])


np.save("./data/train_sentences.npy", train_sentences)
np.save("./data/test_sentences.npy", test_sentences)
np.save("./data/train_labels.npy", train_labels)
np.save("./data/test_labels.npy", test_labels)





