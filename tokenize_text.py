from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from preprocessing import *
import numpy as np

maxlen = 100  # cuts off reviews after 100 words
training_samples = 200 # Trains on 200 samples
validation_samples = 10000 # Validates on 10,000 samples
max_words = 1000 # Considers only the top 10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_text(texts)
sequences = tokenizer.text_to_sequences(texts)
word_index = tokenizer.word_index
print("Found {} unique tokens.".format(len(word_index)))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.array(labels)

print("Shape of data tensor: {}.".format(data.shape))
print("Shape of label tensor: {}.".format(labels.shape))

# Split the data into a training set and a validation set,
# but first shuffles the data, because you're starting with
# data in which samples are ordered (all negative first, then
# all positive)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = data[indices]

X_train = data[:training_samples]
y_train = labels[:training_samples]
X_val = data[training_samples: training_samples+validation_samples]
y_val = data[training_samples: training_samples+validation_samples]