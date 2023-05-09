import re
import os
import sys

import nltk
import numpy as np
from flask import Flask, request, jsonify
from gensim.models.keyedvectors import KeyedVectors
from keras import backend as K
from keras.models import Sequential, load_model
from nltk.corpus import stopwords

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from PyQt5 import QtCore
from PyQt5.QtCore import (QTimer, QAbstractTableModel)
from PyQt5.QtCore import pyqtSlot, QThread, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi


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


class AutomaticEssayScoring(QMainWindow):
    def __init__(self):
        super(AutomaticEssayScoring, self).__init__()

        loadUi("AutomaticEssayScoring.ui", self)

        self.score_pushButton.clicked.connect(self.score_Function)

    @pyqtSlot()
    def score_Function(self):
        input_text = self.plainTextEdit.toPlainText()

        if input_text:
            score = convertToVec(input_text)
            self.score_label.setText(score)
        else:
            QMessageBox.warning(self, 'Status', 'Please Enter some text.', QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = AutomaticEssayScoring()
    window.show()
    sys.exit(app.exec_())
