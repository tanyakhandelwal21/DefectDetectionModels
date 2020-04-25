#import statements from ATT model

from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

from bs4 import BeautifulSoup

import sys
import os
os.environ['KERAS_BACKEND']='tensorflow'

from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence

from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Attention

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers

#import from OGmodel
import pickle
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

MAX_SENT_LENGTH = 100
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

with open('data/y_train.pickle', 'rb') as handle:
    Y_train = pickle.load(handle)
with open('data/y_test.pickle', 'rb') as handle:
    Y_test = pickle.load(handle)
with open('data/y_valid.pickle', 'rb') as handle:
    Y_valid = pickle.load(handle)

with open('data/x_train.pickle', 'rb') as handle:
    X_train = pickle.load(handle)
with open('data/x_test.pickle', 'rb') as handle:
    X_test = pickle.load(handle)
with open('data/x_valid.pickle', 'rb') as handle:
    X_valid = pickle.load(handle)
with open('data/vocab_set.pickle', 'rb') as handle:
    vocabulary_set = pickle.load(handle)

X_train = X_train[:50000]
Y_train = Y_train[:50000]
X_test = X_test[:25000]
Y_test = Y_test[:25000]
X_valid = X_valid[:25000]
Y_valid = Y_valid[:25000]

# Encode training, valid and test instances
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

# Model Definition


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim, )), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        self.tw = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

inputs = Input(shape=(None,), dtype='int32')
#2 layers
embedding = Embedding(encoder.vocab_size, 128)(inputs)
l_lstm = Bidirectional(LSTM(64, return_sequences=True))(embedding)
att = Attention()([embedding,l_lstm])
l_lstm2 = Bidirectional(LSTM(64))(att)

#4 layers
att2 = Attention()([att,l_lstm2])
l_lstm3 = Bidirectional(LSTM(64))(att2)
att3 = Attention()([att2,l_lstm3])
l_lstm4 = Bidirectional(LSTM(64))(att3)

#6 layers
att4 = Attention()([att3,l_lstm4])
l_lstm5 = Bidirectional(LSTM(64))(att4)
att5 = Attention()([att4,l_lstm5])
l_lstm6 = Bidirectional(LSTM(64))(att5)

preds = Dense(1, activation='sigmoid')(l_lstm6)
model = Model(inputs, preds)

#model = tf.keras.Sequential([
#    tf.keras.layers.Embedding(encoder.vocab_size, 64),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#    tf.keras.layers.Dense(64, activation='relu'),
#    tf.keras.layers.Dropout(0.5),
#    tf.keras.layers.Dense(1, activation='sigmoid')
#])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['acc'])

model.summary()
batch_size = 16

# Building generators
class CustomGenerator(Sequence):
    def __init__(self, text, labels, batch_size, num_steps=None):
        self.text, self.labels = text, labels
        self.batch_size = batch_size
        self.len = np.ceil(len(self.text) / float(self.batch_size)).astype(np.int64)
        if num_steps:
            self.len = min(num_steps, self.len)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        batch_x = self.text[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


train_gen = CustomGenerator(X_train, Y_train, batch_size)
valid_gen = CustomGenerator(X_valid, Y_valid, batch_size)
test_gen = CustomGenerator(X_test, Y_test, batch_size)




# Training the model
checkpointer = ModelCheckpoint('data/models/model-{epoch:02d}-{val_loss:.5f}.hdf5',
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=True,
                               mode='min')

callback_list = [checkpointer] #, , reduce_lr
his1 = model.fit_generator(
                    generator=train_gen,
                    epochs=1,
                    validation_data=valid_gen)
                    
                    
                    
                    
predIdxs = model.predict_generator(test_gen, verbose=1)

fpr, tpr, _ = roc_curve(Y_test, predIdxs)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.savefig('auc_model.png')
