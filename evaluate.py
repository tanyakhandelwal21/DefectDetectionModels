import pickle
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Calculating buggy rate for training
with open('data/y_train.pickle', 'rb') as handle:
	Y_train = pickle.load(handle);

train_total = len(Y_train);
train_positive = 0;
for i in Y_train:
	if i == 1:
		train_positive = train_positive + 1;

train_buggy_rate = train_positive/train_total;

print ("Train positive : " + str(train_positive));
print ("Train total : " + str(train_total));
print ("Train buggy rate : " + str(train_buggy_rate));

# Calculating the buggy rate for validation
with open('data/y_valid.pickle', 'rb') as handle:
	Y_valid = pickle.load(handle);

valid_total = len(Y_valid);
valid_positive = 0;
for i in Y_valid:
	if i == 1:
		valid_positive = valid_positive + 1;
	
valid_buggy_rate = valid_positive/valid_total;

print ("Valid positive : " + str(valid_positive));
print ("Valid total : " + str(valid_total));
print ("Valid buggy rate : " + str(valid_buggy_rate));

# Calculating the buggy rate for test
with open('data/y_test.pickle', 'rb') as handle:
	Y_test = pickle.load(handle);

test_total = len(Y_test);
test_positive = 0;
for i in Y_test:
	if i == 1:
		test_positive = test_positive + 1;

test_buggy_rate = test_positive/test_total;

print("Test positive : " + str(test_positive));
print("Test total : " + str(test_total));
print("Test buggy rate : " + str(test_buggy_rate));
