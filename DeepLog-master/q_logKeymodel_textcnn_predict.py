 # -*- coding: utf-8 -*
import torch
import torch.nn as nn
import time
import argparse
import torch.nn.functional as F
from q_logkey_textcnn_train import CNNClassifier
from collections import Counter

# Device configuration
device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters


window_size = 10
num_classes = 29
vocab_size = num_classes + 1
num_candidates = 9
embedding_dim = 128
kernel_dim=150
kernel = (2,3,4)
#kernel = (1,2,3,4)

#model_path = 'model/min_lossAdam_batch_size=2048;epoch=200kernel_dim=150embedding_dim=256.pt'
#model_path = 'model/min_lossbest.pt'
#model_path ='model/199_1234_best'
#model_path ='model_q/0915/test_0915_0.00010787763265403956' #F1-measure: 97.456%
model_path ='model_q/0915_output_20/test_0915_0.00010416382784748743'# F1-measure: 97.481%
#model_path ='model_q/0915_output_30/test_0915_0.00010321662813761'#F1-measure: 97.398%
#model_path ='model_q/0915_output_40/test_0915_0.00010183580255738779'

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

                #seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size).to(device)
                seq = seq.to(torch.int64) 

                qua = torch.tensor(Quantitative_pattern, dtype=torch.float).view(-1, num_classes+1).to(device)
                qua = qua.to(torch.int64) 

                output = model(seq,qua)

                label = torch.tensor(label).view(-1).to(device)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    break

    count = 0
    with torch.no_grad():
        for line in test_normal_loader:
            count+=1
            if count %1000 ==0:
                print("count={} FP={}".format(count,FP))
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                Quantitative_pattern = [0] *  (num_classes+1)
                log_counter = Counter(line[i:i + window_size])
                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]                

                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size).to(device)
                seq = seq.to(torch.int64) 

                qua = torch.tensor(Quantitative_pattern, dtype=torch.float).view(-1, num_classes+1).to(device)
                qua = qua.to(torch.int64) 

                output = model(seq,qua)

                label = torch.tensor(label).view(-1).to(device)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1
                    break











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
