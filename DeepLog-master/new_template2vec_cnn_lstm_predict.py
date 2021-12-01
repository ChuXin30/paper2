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

# Device configuration
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys


'''
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
embedding_dim = 400
#embedding_dim = 600 #top6

#model_path = 'new_template2vec/model/1_0.00011228370781681964'#97.716
#model_path = 'new_template2vec/model/1_0.0001122019873853019'#97.807%
#model_path = 'new_template2vec/model/1_0.00011147994606071304'#
#model_path = 'new_template2vec/model_window_20/1_0.00014059179636279932'#
#model_path = 'new_template2vec/model_top1/1_0.00010986320894377499_234'#top1
#model_path = 'new_template2vec/model_top1/1_0.00011007774247319482_200'#top1
#model_path = 'new_template2vec/model_top2/1_0.00011058248311553153_243'#top2 97.761%
#model_path = 'new_template2vec/model_top2/1_0.00011194246338697383_228'
#model_path = 'new_template2vec/model_top6/1_0.00011178133504338246_244'#top6
#model_path = 'new_template2vec/model_top4_tfidf/1_0.00011037875776705534_217'#top4 tfidf

#event_output_file = 'new_template2vec/event_vector_top3.txt'#top3
#event_output_file = 'new_template2vec/event_vector_top1.txt'#top1
#event_output_file = 'new_template2vec/event_vector_top2.txt'#top2
#event_output_file = 'new_template2vec/event_vector_top6.txt'#top6
#event_output_file = 'new_template2vec/event_vector_top4_tfidf.txt'#top4 tfidf
'''

'''
#noise 20
window_size = 10
kernel_dim=150
num_classes = 30
#num_classes = 29

vocab_size = num_classes + 1
num_candidates = 8
kernel = (2,3,4)
embedding_dim = 400
event_output_file = 'new_template2vec/event_vector_top4_tfidf.txt'#top4 tfidf
model_path = 'data_noise/cnn_lstm_model/1_0.00010972597850404448_496'

#hdfs_test_normal = 'data_noise/noise_0/WITH_S_test_normal'#false positive (FP): 235, false negative (FN): 5717, Precision: 97.931%, Recall: 66.047%, F1-measure: 78.889%
#hdfs_test_abnormal = 'data_noise/noise_0/WITH_S_test_abnormal'

#hdfs_test_normal = 'data_noise/noise_5/WITH_S_test_normal'#false positive (FP): 291, false negative (FN): 5723, Precision: 97.449%, Recall: 66.011%, F1-measure: 78.707%
#hdfs_test_abnormal = 'data_noise/noise_5/WITH_S_test_abnormal'
'''

#cnn_lstm_model_pramenters
'''
#window_size = 8
#window_size = 9
#window_size = 10
#window_size = 11
#window_size = 12

#hidden_size = 64
#hidden_size = 64
#hidden_size = 64
#hidden_size = 64

#num_layers = 2

#kernel_dim = 150
#kernel_dim = 50
#kernel_dim = 100
#kernel_dim = 200

#kernel = (2,3,4)
#kernel = (2,3)
#kernel = (3,4)
#kernel = (3,4,5)


input_size = 1
num_classes = 29
#vocab_size = num_classes + 1
#num_candidates = 8
embedding_dim = 400
event_output_file = 'new_template2vec/event_vector_top4_tfidf.txt'#top4 tfidf

hdfs_test_normal = 'data/hdfs_test_normal'
hdfs_test_abnormal = 'data/hdfs_test_abnormal'
#model_path = 'cnn_lstm_model_pramters/window_size_8/1_0.00012727579072301803_252'#false positive (FP): 522, false negative (FN): 1046, Precision: 96.800%, Recall: 93.788%, F1-measure: 95.270%
#model_path = 'cnn_lstm_model_pramters/window_size_9/1_0.00010369305113004892_250'#false positive (FP): 561, false negative (FN): 888, Precision: 96.602%, Recall: 94.726%, F1-measure: 95.655%
#model_path = 'cnn_lstm_model_pramters/window_size_10/1_0.0001124680221432713_258'#false positive (FP): 712, false negative (FN): 257, Precision: 95.883%, Recall: 98.474%, F1-measure: 97.161%
#model_path = 'cnn_lstm_model_pramters/window_size_11/1_0.00011783055675395498_269'#false positive (FP): 455, false negative (FN): 267, Precision: 97.328%, Recall: 98.414%, F1-measure: 97.868%
#model_path = 'cnn_lstm_model_pramters/window_size_12/1_0.00015016929286274547_272'#false positive (FP): 650, false negative (FN): 134, Precision: 96.254%, Recall: 99.204%, F1-measure: 97.707%
#model_path = 'cnn_lstm_model_pramters/window_size_10/1_0.00010852470714048998_222'#false positive (FP): 585, false negative (FN): 258, Precision: 96.592%, Recall: 98.468%, F1-measure: 97.521%
#model_path = 'cnn_lstm_model_pramters/window_size_10/1_0.0001094490142516148_226'#false positive (FP): 681, false negative (FN): 252, Precision: 96.056%, Recall: 98.503%, F1-measure: 97.264%

#hdfs_test_normal = 'data/hdfs_test_normal_simple'
#hdfs_test_abnormal = 'data/hdfs_test_abnormal_simple'
#model_path = 'cnn_lstm_model_pramters/window_size_10(1)/1_0.0001110755520614321_190'#false positive (FP): 3, false negative (FN): 5, Precision: 98.204%, Recall: 97.041%, F1-measure: 97.619%
#model_path = 'cnn_lstm_model_pramters/window_size_10(1)/1_0.00010821115599482276_230'#false positive (FP): 5, false negative (FN): 5, Precision: 97.041%, Recall: 97.041%, F1-measure: 97.041%
#model_path = 'cnn_lstm_model_pramters/window_size_10(1)/1_0.0001110755520614321_190'
'''
#noise cnn_lstms
input_size = 1
num_classes = 29
embedding_dim = 400
event_output_file = 'new_template2vec/event_vector_top4.txt'
#hdfs_test_normal = 'data_noise/drain_noise_0/WITH_S_test_normal'
#hdfs_test_abnormal = 'data_noise/drain_noise_0/WITH_S_test_abnormal'
vocab_size = num_classes + 1
hdfs_test_normal = 'data_noise/train_10/WITH_S_test_normal'
hdfs_test_abnormal = 'data_noise/train_10/WITH_S_test_abnormal'
def generate(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    #hdfs = set()
    hdfs = []
    with open(name, 'r') as f:
        for line in f.readlines():
            line = list( map(int, line.strip().split()))
         #   line = line + [-1] * (window_size + 1 - len(line))
            line = line + [0] * (window_size + 1 - len(line))
            #hdfs.add(tuple(line))
            hdfs.append(tuple(line))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs
class CNNClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim, kernel_sizes, dropout=0.5):
        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding2 = nn.Embedding(15, 20)

        # print(self.embedding)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])

        self.lstm1 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.fc_cnn_lstm = nn.Linear(len(kernel_sizes) * (kernel_dim) + hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs1, inputs2):
        #print(inputs1.shape)#torch.Size([2048, 10])

      #  inputs1 = self.embedding(inputs1).unsqueeze(1)  # (B,1,T,D)#torch.Size([2048, 1, 10, 128])
       # inputs1 = self.embedding(inputs1)#torch.Size([2048, 10, 128])

        #print(inputs1.shape)#torch.Size([2048, 10, 300])
        inputs1 = inputs1.unsqueeze(1).to(device)
        #print(inputs1.shape)#torch.Size([2048, 1, 10, 300])
        inputs1 = [F.relu(conv(inputs1)).squeeze(3).to(device) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        #print(inputs1.shape)
        inputs1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs1]  # [(N,Co), ...]*len(Ks)
       # print(inputs1.shape)
        concated1 = torch.cat(inputs1, 1)  # torch.Size([2048, 450])
       # print(concated1.shape)#torch.Size([2048, 450])
        #print(inputs2.shape)#torch.Size([2048, 30, 1])
        #print(inputs2.size(0))#2048
        h0_1 = torch.zeros(num_layers, inputs2.size(0),hidden_size).to(device)
        #print(h0_1.shape)#torch.Size([2, 2048, 64])
        c0_1 = torch.zeros(num_layers, inputs2.size(0),hidden_size).to(device)
        #print(c0_1.shape)#torch.Size([2, 2048, 64])
        out1, _ = self.lstm1(inputs2.to(device), (h0_1, c0_1))
        #print(out1.shape)#torch.Size([2048, 30, 64])

        #multi_out = torch.cat((out1[:, -1, :],out1[:, -1, :]), -1)
        #print(out1[:, -1, :].shape)#torch.Size([2048, 64])
        multi_out = out1[:, -1, :] # only need lstm head
        concated = torch.cat((concated1, multi_out), 1)
        #print(concated.shape) #torch.Size([2048, 514])

        concated = self.dropout(concated)
        out = self.fc_cnn_lstm(concated)
        return F.log_softmax(out, 1)

if __name__ == '__main__':
    #print(len(sys.argv))
    #print(sys.argv)

    model_path = sys.argv[1]
    num_layers = int(sys.argv[2])
    hidden_size = int(sys.argv[3])
    kernel_dim = int(sys.argv[4])
    kernel = tuple(map(int,sys.argv[5].split(' ')))
    window_size = int(sys.argv[6])
    num_candidates = int(sys.argv[7])
    vocab_size = num_classes + 1

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



    test_normal_loader = generate(hdfs_test_normal)
    test_abnormal_loader = generate(hdfs_test_abnormal)
    TP = 0
    FP = 0
    # Test the model
    start_time = time.time()

    # abnormal
    count = 0
    with torch.no_grad():
        for line in test_abnormal_loader:
            count += 1
            if count % 1000 == 0:
                print("count={} TP={}".format(count, TP))
           # print(line)
            flag = 0
            for i in range(len(line) - window_size):
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
                        Semantic_pattern.append([0] * embedding_dim)
                    else:
                        Semantic_pattern.append(event2semantic_vec[str(event - 1)])
                # seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)

                # seq = torch.tensor(Semantic_pattern, dtype=torch.float).view(-1, window_size).to(device)
                # seq = seq.to(torch.int64)
                Semantic_pattern = np.array(Semantic_pattern)
                # print(Semantic_pattern.shape)
                # seq = []
                # seq.append(Semantic_pattern)
                seq = torch.tensor(Semantic_pattern, dtype=torch.float).unsqueeze(0)
                # print(seq.shape)
                # qua = torch.tensor(Quantitative_pattern, dtype=torch.float)
                # print(qua.shape)#torch.Size([30])
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
                    flag = 1
                    break
            #print(flag)
    #print(TP)

    #normal
    count = 0
    with torch.no_grad():
        for line in test_normal_loader:
            count += 1
            if count % 10000 == 0:
                print("count={} FP={}".format(count, FP))
            for i in range(len(line) - window_size):
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
                        Semantic_pattern.append([0] * embedding_dim)
                    else:
                        Semantic_pattern.append(event2semantic_vec[str(event - 1)])
                Semantic_pattern = np.array(Semantic_pattern)
                seq = torch.tensor(Semantic_pattern, dtype=torch.float).unsqueeze(0)

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


    #print("count={} TP={}".format(count,TP))
    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
