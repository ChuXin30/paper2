 # -*- coding: utf-8 -*
import torch
import torch.nn as nn
import time
import argparse
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
from gensim.models import KeyedVectors
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
window_size = 5
input_size = 1
hidden_size = 64
num_layers = 1
num_classes = 377
num_epochs = 349
batch_size = 2048
vocab_size = num_classes + 1
embedding_dim = 400
model_dir = 'propose_model/dataset_1'
log = '1_'
kernel_dim = 80
kernel = (2,3,4)
event_output_file = 'data/event_vector_top3.txt'
test_normal_dataset =  'data/dataset_1/test_new'
test_abnormal_dataset =  'data/dataset_1/abnormal_new'
num_candidates = 20

#model_path = 'propose_model/dataset_10/1_9.136864421049494e-05_18'
#model_path='propose_model/dataset_1/1_2.3481708545924974e-05_147'#0.6954919476166931
model_path='propose_model/dataset_1/1_2.1956535596358434e-05_201'
'''
#dataset_80_window_20
'''
window_size = 20
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 377
num_epochs = 349
batch_size = 2048
vocab_size = num_classes + 1
embedding_dim = 400
model_dir = 'propose_model/dataset_80_window_20'
log = '1_'
kernel_dim = 150
kernel = (2,3,4)

event_output_file = 'data/logkey2vec_4.txt'
train_dataset =  'data/dataset_80_window_20/train'
test_normal_dataset =  'data/dataset_80/test_normal'
test_abnormal_dataset =  'data/dataset_80/test_abnormal'

#num_candidates = 200
#model_path='propose_model/dataset_80_window_20/1_8.630436851580261e-05_26'#false positive (FP): 16667, false negative (FN): 6351, Precision: 39.913%, Recall: 63.546%, F1-measure: 49.030%
num_candidates = 250
model_path='propose_model/dataset_80_window_20/1_8.630436851580261e-05_26'#false positive (FP): 11055, false negative (FN): 6561, Precision: 49.557%, Recall: 62.341%, F1-measure: 55.219%
'''
#order_random_dataset_20
'''
window_size = 20
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 377
num_epochs = 349
batch_size = 2048
vocab_size = num_classes + 1
embedding_dim = 400
model_dir = 'propose_model/dataset_80_window_20'
log = '1_'
kernel_dim = 80
kernel = (2,3,4)

event_output_file = 'data/logkey2vec_4.txt'
train_dataset =  'data/order_random_dataset_20/train'
test_normal_dataset =  'data/order_random_dataset_20/test_normal'
test_abnormal_dataset =  'data/order_random_dataset_20/test_abnormal'

model_path='propose_model/order_random_dataset_20/1_0.0001970409512153388_167'
#num_candidates = 150#count=1000 FP=1false positive (FP): 128, false negative (FN): 6500, Precision: 98.926%, Recall: 64.467%, F1-measure: 78.063%
num_candidates = 100#count=1000 FP=2 count=2000 FP=3 count_ab=1000 TP=704
#num_candidates = 50#count=1000 FP=5 count=2000 FP=8 count_ab=1000 TP=704
#num_candidates = 10#count=1000 FP=18 count=4000 FP=70 count_ab=1000 TP=705
#num_candidates = 5#count=1000 FP=30 count_ab=1000 TP=717
'''

#drain_random_with_level_20
'''
window_size = 20
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 1916
num_layers = 2
batch_size = 2048
vocab_size = num_classes + 1
embedding_dim = 400
log = '1_'
kernel_dim = 80
kernel = (2,3,4)
event_output_file = 'data/drain_random_with_level_20/event_vector_top4.txt'
test_normal_dataset = 'data/drain_random_with_level_20/test_normal_simple'
test_abnormal_dataset='data/drain_random_with_level_20/test_abnormal_simple'
model_path = 'propose_model/drain_random_with_level_20/1_0.0011936919726184597_10'#
#num_candidates = 500#false positive (FP): 15, false negative (FN): 30, Precision: 98.333%, Recall: 96.721%, F1-measure: 97.521%
#num_candidates = 800#false positive (FP): 13, false negative (FN): 157, Precision: 98.314%, Recall: 82.842%, F1-measure: 89.917%
#num_candidates = 300#false positive (FP): 18, false negative (FN): 1, Precision: 98.069%, Recall: 99.891%, F1-measure: 98.971%
#num_candidates = 200#false positive (FP): 20, false negative (FN): 1, Precision: 97.859%, Recall: 99.891%, F1-measure: 98.864%
#num_candidates = 100#false positive (FP): 42, false negative (FN): 1, Precision: 95.607%, Recall: 99.891%, F1-measure: 97.702%
test_normal_dataset = 'data/drain_random_with_level_20/test_normal'
test_abnormal_dataset='data/drain_random_with_level_20/test_abnormal'
num_candidates = 300#
'''
# #noise
# window_size = 20
# input_size = 1
# hidden_size = 64
# num_layers = 2
# num_classes = 1916
# num_layers = 2
# batch_size = 2048
# vocab_size = num_classes + 1
# embedding_dim = 300
# log = '1_'
# kernel_dim = 100
# kernel = (2,3,4)
# event_output_file = 'noise/event_vector_top2_with_level.txt'
# #test_normal_dataset = 'noise/drain_random_with_level_20/test_normal_simple'
# #test_abnormal_dataset='noise/drain_random_with_level_20/test_abnormal_simple'
# #model_path = 'noise/our_model_random/1_0.0017579061818849039_104'#0.94
# #model_path = 'noise/our_model_random/1_0.0022038982871419015_14'#300 0.98
# #model_path = 'noise/our_model_random/1_0.002318691305375015_10'#
# #test_normal_dataset = 'noise_2/noise_0/WITH_S_test_normal_simple'
# #test_abnormal_dataset='noise_2/noise_0/WITH_S_test_abnormal_simple'
# #num_candidates = 300#
window_size = 20
input_size = 1
#hidden_size = 64
# hidden_size = 128
num_classes = 1916
# num_layers = 2
# batch_size = 2048
vocab_size = num_classes + 1
#embedding_dim = 500
# embedding_dim = 400
log = '1_'
# kernel_dim = 100
# kernel_dim = 80
# kernel = (2,3,4)
# kernel = (2,3)
#event_output_file = 'noise/event_vector_top4_with_level.txt'
# event_output_file = 'noise/event_vector_top4.txt'


def generate(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    #hdfs = set()
    hdfs = []
    with open(name, 'r') as f:
        line_all = ''
        for line in f.readlines():
            line_all += ' '+ line.replace('\n','')
        line_all = tuple(map(int, line_all.strip().split()))
        hdfs.append(tuple(line_all))
        '''
        for line in f.readlines():
            line = list( map(int, line.strip().split()))
            line = line + [29] * (window_size + 1 - len(line))

            #hdfs.add(tuple(line))
            hdfs.append(tuple(line))
        '''
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    print("len=%d"%(len(hdfs[0])))
    return hdfs

class CNNClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim, kernel_sizes, dropout=0.5):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding2 = nn.Embedding(15, 20)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])

        self.lstm1 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.fc_cnn_lstm = nn.Linear(len(kernel_sizes) * (kernel_dim) + hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs1, inputs2):
        inputs1 = inputs1.unsqueeze(1).to(device)
        inputs1 = [F.relu(conv(inputs1)).squeeze(3).to(device) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        inputs1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs1]  # [(N,Co), ...]*len(Ks)
        concated1 = torch.cat(inputs1, 1)  # torch.Size([2048, 450])
        h0_1 = torch.zeros(num_layers, inputs2.size(0),hidden_size).to(device)
        c0_1 = torch.zeros(num_layers, inputs2.size(0),hidden_size).to(device)
        out1, _ = self.lstm1(inputs2.to(device), (h0_1, c0_1))
        multi_out = out1[:, -1, :] # only need lstm head
        concated = torch.cat((concated1, multi_out), 1)
        concated = self.dropout(concated)
        out = self.fc_cnn_lstm(concated)
        return F.log_softmax(out, 1)


if __name__ == '__main__':
    num_candidates = int(sys.argv[1])
    model_path = sys.argv[2]
    test_normal_dataset = sys.argv[3]
    test_abnormal_dataset = sys.argv[4]
    hidden_size = int(sys.argv[5])
    num_layers = int(sys.argv[6])
    kernel = tuple(map(int,sys.argv[7].split(' ')))
    kernel_dim = int(sys.argv[8])
    embedding_dim = int(sys.argv[9])
    event_output_file = sys.argv[10]

    model = CNNClassifier(vocab_size, embedding_dim, num_classes, kernel_dim, kernel, 0.5).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    event2semantic_vec = KeyedVectors.load_word2vec_format(event_output_file, binary=False)

    print('-----------------------------cnn_lstm---------------------------------')
    print('model_path: {}'.format(model_path))
    print('window_size:{}'.format(window_size))
    print('num_classes:{}'.format(num_classes))
    print('vocab_size:{}'.format(vocab_size))
    print('num_candidates:{}'.format(num_candidates))
    print('embedding_dim:{}'.format(embedding_dim))
    print('kernel_dim:{}'.format(kernel_dim))
    print('kernel:{}'.format(kernel))
    print('num_layers:{}'.format(num_layers))
    print('hidden_size:{}'.format(hidden_size))
    # print('log:{}'.format(log))
    print('event_output_file:{}'.format(event_output_file))
    print('test_normal_dataset:{}'.format(test_normal_dataset))
    print('test_abnormal_dataset:{}'.format(test_abnormal_dataset))


    test_normal_loader = generate(test_normal_dataset)
    test_abnormal_loader = generate(test_abnormal_dataset)
    TP = 0
    FP = 0
    # Test the model
    start_time = time.time()

    # abnormal
    count_ab = 0
    with torch.no_grad():
        for line in test_abnormal_loader:
            for i in range(len(line) - window_size):
                if i % window_size == 0:
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    Quantitative_pattern = [0] * (num_classes + 1)
                    log_counter = Counter(line[i:i + window_size])
                    # print(log_counter)
                    for key in log_counter:
                        Quantitative_pattern[key] = log_counter[key]

                    Sequential_pattern = line[i:i + window_size]
                    Semantic_pattern = []
                    for event in Sequential_pattern:
                        if event == 0:
                            Semantic_pattern.append(event2semantic_vec[str(event)])
                        else:
                            Semantic_pattern.append(event2semantic_vec[str(event)])
                    Semantic_pattern = np.array(Semantic_pattern)
                    seq = torch.tensor(Semantic_pattern, dtype=torch.float).unsqueeze(0)

                    q = []
                    Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
                    q.append(Quantitative_pattern)
                    q = torch.tensor(q, dtype=torch.float)
                    output = model(seq, q)
                    count_ab += 1
                    if count_ab % 1000 == 0:
                        print("count_ab={} TP={}".format(count_ab, TP))
                    label = torch.tensor(label).view(-1).to(device)
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]
                    if label not in predicted:
                        TP += 1
                        continue
    print(TP)

    # normal
    count_nor = 0
    with torch.no_grad():
        for line in test_normal_loader:
            # print(line)
            for i in range(len(line) - window_size):
                if i % window_size == 0:
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    Quantitative_pattern = [0] * (num_classes + 1)
                    log_counter = Counter(line[i:i + window_size])
                    for key in log_counter:
                        Quantitative_pattern[key] = log_counter[key]

                    Sequential_pattern = line[i:i + window_size]
                    Semantic_pattern = []
                    for event in Sequential_pattern:
                        if event == 0:
                            Semantic_pattern.append(event2semantic_vec[str(event)])
                        else:
                            Semantic_pattern.append(event2semantic_vec[str(event)])
                    Semantic_pattern = np.array(Semantic_pattern)
                    seq = torch.tensor(Semantic_pattern, dtype=torch.float).unsqueeze(0)

                    q = []
                    Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
                    q.append(Quantitative_pattern)
                    q = torch.tensor(q, dtype=torch.float)

                    output = model(seq, q)

                    label = torch.tensor(label).view(-1).to(device)
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]
                    count_nor += 1
                    if count_nor % 1000 == 0:
                        print("count={} FP={}".format(count_nor, FP))
                    if label not in predicted:
                        FP += 1
                        continue



    print("count={} TP={}".format(count_ab,TP))
    # Compute precision, recall and F1-measure
    #FN =  count  - TP

    FN = count_ab - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
