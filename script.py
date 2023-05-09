import re
import os
import nltk
import numpy as np
from flask import Flask, request, jsonify
from gensim.models.keyedvectors import KeyedVectors
from keras import backend as K
from keras.models import Sequential, load_model
from nltk.corpus import stopwords

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def sent2word(x):
    stop_words = set(stopwords.words('english'))
    x = re.sub("[^A-Za-z]", " ", x)
    x.lower()
    filtered_sentence = []
    words = x.split()
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words = []
    for i in raw:
        if len(i) > 0:
            final_words.append(sent2word(i))
    return final_words


def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.index_to_key)
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec, model[i])
    vec = np.divide(vec, noOfWords)
    return vec


def getVecs(essays, model, num_features):
    c = 0
    essay_vecs = np.zeros((len(essays), num_features), dtype="float32")
    for i in essays:
        essay_vecs[c] = makeVec(i, model, num_features)
        c += 1
    return essay_vecs


def get_model():
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4,
              input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop', metrics=['mae'])
    model.summary()
    return model


def convertToVec(text):
    content = text
    if len(content) > 20:
        num_features = 300
        model = KeyedVectors.load_word2vec_format(
            "word2vecmodel.bin", binary=True)
        clean_test_essays = [sent2word(content)]
        testDataVecs = getVecs(clean_test_essays, model, num_features)
        testDataVecs = np.array(testDataVecs)
        testDataVecs = np.reshape(
            testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

        lstm_model = load_model("final_lstm.h5")
        preds = lstm_model.predict(testDataVecs)
        return str(round(preds[0][0]))


final_text = "Well computers can be a good or a bad thing. I don'@CAPS1 realy see computers can be a bad thing for me. I also know computers can or will help people all around the world. I think computers has positive effects on people like me. Computers teaches hand-eye coordination. It can help if you need to find out reasearch for a school project. You can create lots of things on computers like music, desiner logos, banners and lots of other creative things. With computer you can look up available homes and apartments. You can even go online and fill out a job application and save trips to stores cool is that!! Well im a regular person not rich not famous but computers provide lot of information people use today. Thats why I think that computers has a positive effects on people and you don'@CAPS1 have to be super smart to use one."

score = convertToVec(final_text)
print("------------- ESSAY SCORE -----------------\n")
print(score)
print("\n----------------- END ---------------------\n")
