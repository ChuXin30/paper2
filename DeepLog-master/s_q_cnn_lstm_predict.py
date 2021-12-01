 # -*- coding: utf-8 -*
import torch
import torch.nn as nn
import time
import argparse
import torch.nn.functional as F
from s_q_cnn_lstm_train import CNNClassifier,read_json
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json

# Device configuration
device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
window_size = 10
num_classes = 29
vocab_size = num_classes + 1
num_candidates = 8
embedding_dim = 300
kernel_dim=150
kernel = (2,3,4)
model_path = 's_q_cnn_lstm/model_loganomaly/0.00011153799964098026'
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    window_size = args.window_size
    model = CNNClassifier(vocab_size, embedding_dim, num_classes, kernel_dim, kernel, 0.5).to(device)

   # model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()


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

    event2semantic_vec = read_json('s_q_cnn_lstm/event2semantic_vec.json')

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

                Sequential_pattern = line[i:i + window_size]
                Semantic_pattern = []
                for event in Sequential_pattern:
                    if event == 0:
                        Semantic_pattern.append([-1] * 300)
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


    count = 0
    with torch.no_grad():
        for line in test_abnormal_loader:
            count+=1
            if count %1000 ==0:
                print("count={} TP={}".format(count,TP))
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                Quantitative_pattern = [0] * (num_classes+1)
                log_counter = Counter(line[i:i + window_size])
                #print(log_counter)
                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]


                Sequential_pattern = line[i:i + window_size]
                Semantic_pattern = []
                for event in Sequential_pattern:
                    if event == 0:
                        Semantic_pattern.append([-1] * 300)
                    else:
                        Semantic_pattern.append(event2semantic_vec[str(event -1)])
                #seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)

                #seq = torch.tensor(Semantic_pattern, dtype=torch.float).view(-1, window_size).to(device)
                #seq = seq.to(torch.int64)
                Semantic_pattern =  np.array(Semantic_pattern)
                #print(Semantic_pattern.shape)
               # seq = []
               # seq.append(Semantic_pattern)
                seq = torch.tensor(Semantic_pattern, dtype=torch.float).unsqueeze(0)
                #print(seq.shape)
               # qua = torch.tensor(Quantitative_pattern, dtype=torch.float)
                #print(qua.shape)#torch.Size([30])
                q = []
                Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
                q.append(Quantitative_pattern)
                q = torch.tensor(q, dtype=torch.float)
                #print(q.shape)#torch.Size([1, 30, 1])

                output = model(seq,q)

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
