# -*- coding: utf-8 -*-
import io, os, sys
import numpy as np
import pandas as pd
import csv

from typing import Any, Dict, List, Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM, Dropout
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from datasets import load_dataset
import tensorflow as tf
import sklearn.metrics as metrics

def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())

    # return a single tensor value
    return _f1score

def macro_f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def ynat_macro_f1(preds: np.ndarray, targets: np.ndarray) -> Any:
    return metrics.f1_score(targets, preds, average="macro") * 100.0

def train(script, dataset):
    f = open(script, mode='r')
    line=None
    result_list = []
    while line != ' ':
        VEC_FILE_PATH = f.readline().strip()

        if 'vec' not in VEC_FILE_PATH:
            break

        fin = io.open(VEC_FILE_PATH, 'r', encoding='utf-8', newline='\n', errors='ignore')

        if 'stroke' in VEC_FILE_PATH:
            YNAT_TRAIN_PATH = './dataset/decompose_train_stroke.txt'
            YNAT_TEST_PATH = './dataset/decompose_test_stroke.txt'
        elif 'cji' in VEC_FILE_PATH:
            YNAT_TRAIN_PATH = './dataset/decompose_train_cji.txt'
            YNAT_TEST_PATH = './dataset/decompose_test_cji.txt'
        elif 'bts' in VEC_FILE_PATH:
            YNAT_TRAIN_PATH = './dataset/decompose_train_bts.txt'
            YNAT_TEST_PATH = './dataset/decompose_test_bts.txt'
        else:
            YNAT_TRAIN_PATH = './dataset/decompose_train_jm.txt'
            YNAT_TEST_PATH = 'dataset/decompose_val_jm.txt'

        # print("loading...")

        word_vecs = {}

        for i, line in enumerate(fin):
            tokens = line.rstrip().split(' ')
            array = np.array(list(map(float, tokens[1:])))
            array = array / np.sqrt(np.sum(array * array + 1e-8))
            word_vecs[tokens[0]] = array

        x_train = []
        for line in open(YNAT_TRAIN_PATH, mode='r', encoding='utf-8'):
            x_train.append(line.strip())

        y_train = dataset['train']['label']

        x_test = []
        for line2 in open(YNAT_TEST_PATH, mode='r', encoding='utf-8'):
            x_test.append(line2.strip())

        y_test = dataset['validation']['label']

        # print("Tokenizing...")

        tokenizer = Tokenizer(oov_token="<UNK>", filters='!"#$%()*+,/:;<=>?@[\\]_`{|~\t\n')
        tokenizer.fit_on_texts(x_train)
        train_sequence = tokenizer.texts_to_sequences(x_train)
        MAX_SEQUENCE_LENGTH = 10
        train_inputs = pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
        train_labels = np.array(y_train)

        test_sequence = tokenizer.texts_to_sequences(x_test)
        test_inputs = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
        test_labels = np.array(y_test)

        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        word_size = len(tokenizer.word_index) + 1  # 1을 더해주는 것은 padding으로 채운 0 때문입니다
        EMBEDDING_DIM = 300
     
        embedding_matrix = np.zeros((word_size, EMBEDDING_DIM))

        # tokenizer에 있는 단어 사전을 순회하면서 word2vec의 300차원 vector를 가져옵니다
        for word, idx in tokenizer.word_index.items():
            embedding_vector = word_vecs[word] if word in word_vecs else None
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector

        num_classes = 7

        with tf.device('/gpu:0'):
            acc_total = 0
            f1_total = 0
            for seed in [45, 63, 1126]:
                # model1 : label1, pearson
                tf.random.set_seed(seed)

                model = Sequential()
                model.add(
                    Embedding(word_size, 300, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix],
                              trainable=False))
                model.add(LSTM(300, dropout=0.5))
                model.add(Dropout(0.5))
                model.add(Dense(num_classes, activation='softmax'))

                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                model.fit(train_inputs, train_labels, epochs=10, batch_size=64)
                loss, acc = model.evaluate(test_inputs, test_labels)
                pred = model.predict(test_inputs)
                pred_flatten = pred.argmax(axis=1)
                target = test_labels.argmax(axis=1)
                macro_f1 = ynat_macro_f1(pred_flatten, target)

                f1_total += macro_f1
                acc_total += acc

        # _loss, _acc, _precision, _recall, _f1score, _macro_f1 = model.evaluate(test_inputs, test_labels)
        # print(
        #     'loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}, macro_f1: {:.3f}'.format(
        #         _loss, _acc,
        #         _precision,
        #         _recall,
        #         _f1score,
        #         _macro_f1))
        result_list.append([VEC_FILE_PATH, acc_total/3, f1_total/3])

    return result_list


def main():
    # YNAT_TRAIN_PATH = sys.argv[1]
    # YNAT_TEST_PATH = sys.argv[2]
    # VEC_FILE_PATH = sys.argv[3]
    SCRIPT_PATH = sys.argv[1]

    task = "ynat"

    dataset = load_dataset('klue', task)

    result_file = open('./result_train.csv', 'w', newline='')
    wr = csv.writer(result_file)

    result_list = train(SCRIPT_PATH, dataset)

    for res in result_list:
        wr.writerow(res)
    result_file.close()


if __name__ == "__main__":
    main()
