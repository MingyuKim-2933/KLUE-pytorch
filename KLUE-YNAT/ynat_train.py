# -*- coding: utf-8 -*-
import io, sys
import random
import numpy as np
import sklearn.metrics as metrics
from datasets import load_dataset
import csv
from nltk import FreqDist
from model import LSTM
from trainer import Trainer
from dataset import Dataset


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)


def make_encoded_text(tokenized_text, word_to_index):
    encoded_text = []

    for line in tokenized_text:  # 입력 데이터에서 1줄씩 문장을 읽음
        temp = []
        for w in line:  # 각 줄에서 1개씩 글자를 읽음
            try:
                temp.append(word_to_index[w])  # 글자를 해당되는 정수로 변환
            except KeyError:  # 단어 집합에 없는 단어일 경우 unk로 대체된다.
                temp.append(word_to_index['unk'])  # unk의 인덱스로 변환

        encoded_text.append(temp)

    return encoded_text


def make_vocab(text):
    tokenized_text = []

    for sentence in text:
        tokenized_text.append(sentence.split())

    vocab = FreqDist(np.hstack(tokenized_text))
    vocab = vocab.most_common(len(vocab))

    word_to_index = {word[0]: index + 2 for index, word in enumerate(vocab)}
    word_to_index['pad'] = 1
    word_to_index['unk'] = 0

    return vocab, word_to_index


def make_padding(encoded_text, word_to_index, max_len):
    for line in encoded_text:
        if len(line) < max_len:  # 현재 샘플이 정해준 길이보다 짧으면
            line += [word_to_index['pad']] * (max_len - len(line))  # 나머지는 전부 'pad' 토큰으로 채운다.
        elif len(line) > max_len:
            for i in range(len(line) - max_len):
                del line[-1]

    return encoded_text


def get_dataloader(sequence, label, batch_size=64):

    train_dataset = Dataset(sequence, label)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"* 배치 크기: {batch_size}")
    print(f"* 전체 훈련 배치 개수: {len(train_loader)}")

    return train_loader


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
            YNAT_VAL_PATH = './dataset/decompose_val_stroke.txt'
        elif 'cji' in VEC_FILE_PATH:
            YNAT_TRAIN_PATH = './dataset/decompose_train_cji.txt'
            YNAT_VAL_PATH = './dataset/decompose_val_cji.txt'
        elif 'bts' in VEC_FILE_PATH:
            YNAT_TRAIN_PATH = './dataset/decompose_train_bts.txt'
            YNAT_VAL_PATH = './dataset/decompose_val_bts.txt'
        else:
            YNAT_TRAIN_PATH = './dataset/decompose_train_jm.txt'
            YNAT_VAL_PATH = './dataset/decompose_val_jm.txt'

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

        x_val = []
        for line2 in open(YNAT_VAL_PATH, mode='r', encoding='utf-8'):
            x_val.append(line2.strip())

        y_val = dataset['validation']['label']

        vocab, word_to_index = make_vocab(x_train)

        tokenized_x_train = []
        for sentence in x_train:
            tokenized_x_train.append(sentence.split())

        tokenized_x_val = []
        for sentence in x_val:
            tokenized_x_val.append(sentence.split())

        encoded_x_train = make_encoded_text(tokenized_text=tokenized_x_train, word_to_index=word_to_index)
        encoded_x_val = make_encoded_text(tokenized_text=tokenized_x_val, word_to_index=word_to_index)

        max_len = 10

        encoded_x_train = make_padding(encoded_x_train, word_to_index, max_len)
        encoded_x_val = make_padding(encoded_x_val, word_to_index, max_len)

        word_size = len(vocab) + 2
        EMBEDDING_DIM = 300
        embedding_matrix = np.zeros((word_size, EMBEDDING_DIM))


        # tokenizer에 있는 단어 사전을 순회하면서 word2vec의 300차원 vector를 가져옵니다
        for word, idx in word_to_index.items():
            embedding_vector = word_vecs[word] if word in word_vecs else None
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector

        embedding_matrix = torch.FloatTensor(embedding_matrix)
        # print('임베딩 벡터의 개수와 차원 : {} '.format(embedding_matrix.shape))

        hidden_size = 300
        num_layers = 1
        num_classes = 7
        num_epochs = 3
        learning_rate = 1e-3
        batch_size = 64

        train_loader = get_dataloader(encoded_x_train, y_train, batch_size=batch_size)

        total_acc = 0
        total_f1 = 0

        for seed in [42, 63, 1126]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            model = LSTM(weights=embedding_matrix, input_size=EMBEDDING_DIM, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, device=device)
            trainer = Trainer(model, learning_rate, num_epochs, device)

            trainer.train(train_loader)

            val_loader = get_dataloader(encoded_x_val, y_val, batch_size=batch_size)
            acc, f1_score = trainer.evaluation(val_loader, num_classes)
            total_acc += acc
            total_f1 += f1_score

        avg_acc = total_acc / 3
        avg_f1 = total_f1 / 3

        result_list.append([VEC_FILE_PATH, avg_acc, avg_f1])

    return result_list



def main():
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
