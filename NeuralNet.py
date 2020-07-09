import os
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, \
    Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.layers.merge import add
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from numpy import array
from pickle import dump
from keras.callbacks import LearningRateScheduler

from PreNeuralNetTransferLearn import train_descriptions, train_features


# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n = 0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n += 1
            # print('key: ' + key)
            # retrieve the photo feature
            photo = photos[key + '.jpg']
            # print('Photo Shape:')
            # print(photo.shape)
            for desc in desc_list:
                # print('desc: ' + desc)
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # print('here')
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n == num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n = 0


# Create a list of all the training captions
all_train_captions = []
print("Train Descriptions: ")
print(train_descriptions)
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
print(len(all_train_captions))

# Consider only words which occur at least 10 times in the corpus
word_count_threshold = 0
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

# we will represent every unique word in the vocabulary by an integer (index)
ixtoword = {}  # index to word
wordtoix = {}  # word to index

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

dump(wordtoix, open("./pickle/wordtoix.pkl", "wb"))
dump(ixtoword, open("./pickle/ixtoword.pkl", "wb"))
print('DUMPED')

vocab_size = len(ixtoword) + 1  # one for appended 0's
print(vocab_size)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# Load Glove vectors
glove_dir = './Glove'
embeddings_index = {}  # empty dictionary
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 200

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    # if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

# Final Neural Network Architecture

print('---------------Neural Network Training ------------------')

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.summary()
print(model.layers[2])
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False
model.compile(loss='categorical_crossentropy', optimizer='adam')


def scheduler(epoch):
    if epoch < 20:
        return 0.001
    else:
        return 0.0001


callback = LearningRateScheduler(scheduler)

epochs = 30
number_pics_per_batch = 3
steps = len(train_descriptions)  # number_pics_per_bath

print(train_features)
for key, desc_list in train_descriptions.items():
    # retrieve the photo feature
    photo = train_features[key + '.jpg']
    print(photo.shape)

for i in range(epochs):
    if epochs < 20:
        number_pics_per_batch = 3
    else:
        number_pics_per_batch = 6

    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_batch)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1, callbacks=[callback])
    # model.save('./model_weights/model_' + str(i) + '.h5')


model.save('./model_weights/model_30.h5')

