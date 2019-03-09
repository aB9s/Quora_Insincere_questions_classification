# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 23:44:01 2019

@author: aB9
"""

""""""""""""""""" 
 IMPORT LIBRARIES 
"""""""""""""""""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import csv
import os

"""""""""""
 IMPORT DATA 
"""""""""""
# Train Data
train_data = pd.read_csv('data/train.csv')
train_data.shape
train_data.head(10)
train_data.columns


# Test Data
test_data = pd.read_csv('data/test.csv')
test_data.shape
test_data.head(10)
test_data.columns

# lets divide training data into feature set (text) and targets 
X_train, y_train = train_data['question_text'].values, train_data['target'].values
X_test = test_data['question_text'].values

X_train[9]
y_train[9]
X_test[9]


"""""""""""""""""""""
 TEXT PREPROCESSING 
"""""""""""""""""""""
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# REGEXs to remove unwanted patterns from the text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

'''
TEXT_PREPROCESSING for removing unwanted stopwords, symbols and modify text 
	>> text to be processed
	<< modified text
'''
def text_preprocessing(text_):
	
	# convert text to lower case
	text_ = text_.lower()
	
	# replace symbols with a space in the text
	text_ = re.sub(REPLACE_BY_SPACE_RE, " ", text_)
	
	# truncate unwanted symbols from text
	text_ = re.sub(BAD_SYMBOLS_RE,"",text_)
	
	# delete stopwords from text
	text_ = " ".join(word for word in text_.split() if word not in STOPWORDS)
	
	return text_

# Process training data text
X_train_processed = [text_preprocessing(text) for text in X_train]
X_train_processed[9]

# Process testing data text
X_test_processed = [text_preprocessing(text) for text in X_test]
X_test_processed[9]

"""""""""""""""""""""""""""""""""""""""
 CALCULATE FREQUENCY OF WORDS (TOKENS) 
"""""""""""""""""""""""""""""""""""""""
'''
WORDS_FREQ_COUNTS used to calculate frequncy of words in the text corpus
	>> text to be processed
	<< words_freq dictionary of words and their associated frequency
'''
def words_freq_counts(text):
	words_freq = {}
	
	for line in text:
		for word in line.split():
			if word not in words_freq:
				words_freq[word] = 1
			else:
				words_freq[word] +=1
	
	return words_freq

# Training data words frequency count
words_count = words_freq_counts(X_train_processed)

# total couunt of words (tokens)
len(words_count)

# most common terms in text corpora
sorted(words_count.items(), key = lambda x: x[1], reverse = True)[:10]

""""""""""""""""""""""
 BAG OF WORDS APPROACH 
""""""""""""""""""""""
WORDS_TO_INDEX = {b[0]:a for a, b in enumerate(sorted(words_count.items(), key=lambda x:x[1], reverse=True))}
INDEX_TO_WORDS = {b: a for a,b in WORDS_TO_INDEX.items()}
ALL_WORDS = WORDS_TO_INDEX.keys()
DICT_SIZE = 230000

'''
BAG_OF_WORDS_PROCESSING creates a vector of tokens from strings
	>> text to be processed into vector of tokens
	>> words_to_index to refer for the token generation
	>> dict_size default size of all the vectors
	<< vector of tokens corrosponding to the given text string
'''
def bag_of_words_processing(text, words_to_index,dict_size):
	# Create a zero vector equaling the size of words list
	tokenized_vector = np.zeros([dict_size])
	
	for word in set(text.split()):
		if word in words_to_index:
			tokenized_vector[words_to_index[word]] = 1
			
	return tokenized_vector

# Apply BOW appraoch to train anf test dataset
from scipy import sparse as sp_sparse

# Train data
X_train_bow = sp_sparse.vstack([sp_sparse.csr_matrix(bag_of_words_processing(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
print('X-train bow shape:', X_train_bow.shape)

# Test data
X_test_bow = sp_sparse.vstack([sp_sparse.csr_matrix(bag_of_words_processing(text,WORDS_TO_INDEX,DICT_SIZE)) for text in X_test])
print('X-test bow shape:', X_test_bow.shape)


"""""""""""
 PREDICTION 
"""""""""""
# We will be using Stochaistic stochastic gradient descent classifier
from sklearn.linear_model import SGDClassifier

""" BAG OF WORDS APPROACH """
classifier_bow = SGDClassifier(loss='hinge', penalty='l1',n_iter=20)
classifier_bow.fit(X_train_bow, y_train)

# predict
y_pred_bow = classifier_bow.predict(X_test_bow)