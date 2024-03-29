 # -*- coding: utf-8 -*
import torch
import torch.nn as nn
import time
import argparse
import torch.nn.functional as F
from new_template2vec_lstm_train import CNNClassifier
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
from gensim.models import KeyedVectors

# Device configuration
device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
window_size = 10
#window_size = 20
#kernel_dim=150
#kernel_dim=300
kernel_dim=150
num_classes = 29
vocab_size = num_classes + 1
num_candidates = 8
kernel = (2,3,4)

#embedding_dim = 300
#embedding_dim = 100 #top1
#embedding_dim = 200 #top2
#embedding_dim = 400
embedding_dim = 600 #top6


#model_path = 'new_template2vec/model_only_lstm/1_0.00014464962949563404_246'#92.918%
model_path = 'new_template2vec/model_only_lstm/1_0.00014004531689888037_291'#top4 tfidf

event_output_file = 'new_template2vec/event_vector_top4.txt'#top4

def generate(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    #hdfs = set()
    hdfs = []
    with open('data/' + name, 'r') as f:
        for line in f.readlines():
            line = list( map(int, line.strip().split()))
         #   line = line + [-1] * (window_size + 1 - len(line))
            line = line + [29] * (window_size + 1 - len(line))

            #hdfs.add(tuple(line))
            hdfs.append(tuple(line))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs



if __name__ == '__main__':

    model = CNNClassifier(vocab_size, embedding_dim, num_classes, kernel_dim, kernel, 0.5).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    event2semantic_vec = KeyedVectors.load_word2vec_format(event_output_file, binary=False)

    print('model_path: {}'.format(model_path))
    print('window_size:{}'.format(window_size))
    print('num_classes:{}'.format(num_classes))
    print('vocab_size:{}'.format(vocab_size))
    print('num_candidates:{}'.format(num_candidates))
    print('embedding_dim:{}'.format(embedding_dim))
    print('kernel_dim:{}'.format(kernel_dim))
    print('kernel:{}'.format(kernel))



    test_normal_loader = generate('hdfs_test_normal')
    test_abnormal_loader = generate('hdfs_test_abnormal')
    TP = 0
    FP = 0
    # Test the model
    start_time = time.time()



    #normal
    count = 0
    with torch.no_grad():
        for line in test_normal_loader:
            count += 1
            if count % 1000 == 0:
                print("count={} FP={}".format(count, FP))
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                Quantitative_pattern = [0] * (num_classes + 1)
                log_counter = Counter(line[i:i + window_size])
                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]
                seq = []
                q = []
                Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
                q.append(Quantitative_pattern)
                q = torch.tensor(q, dtype=torch.float)
                output = model(seq, q)

                label = torch.tensor(label).view(-1).to(device)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1
                    break

    # abnormal
    count = 0
    with torch.no_grad():
        for line in test_abnormal_loader:
            count += 1
            if count % 1000 == 0:
                print("count={} TP={}".format(count, TP))
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                Quantitative_pattern = [0] * (num_classes + 1)
                log_counter = Counter(line[i:i + window_size])
                # print(log_counter)
                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]

                seq = []
                q = []
                Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
                q.append(Quantitative_pattern)
                q = torch.tensor(q, dtype=torch.float)
                # print(q.shape)#torch.Size([1, 30, 1])

                output = model(seq, q)

                label = torch.tensor(label).view(-1).to(device)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    break
    print(TP)

    print("count={} TP={}".format(count,TP))
    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
