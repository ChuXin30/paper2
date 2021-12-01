import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import torch.nn.functional as F
from collections import Counter
import numpy as np
import json
from gensim.models import KeyedVectors
import  sys
#device = torch.device("cpu") #cuda
device = torch.device("cuda") #cuda

# Hyperparameters
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
#model_dir = 'propose_model/dataset_10'
model_dir = 'propose_model/dataset_1'

log = '1_'
kernel_dim = 80
kernel = (2,3,4)
event_output_file = 'data/event_vector_top3.txt'
train_dataset =  'data/dataset_1/train'
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
log = '1_'
#kernel_dim = 150
kernel = (2,3,4)
event_output_file = 'data/logkey2vec_4.txt'
#train_dataset =  'data/dataset_80/train'
#model_dir = 'propose_model/dataset_80_window_20'

kernel_dim = 80
train_dataset =  'data/order_random_dataset_20/train'
model_dir = 'propose_model/order_random_dataset_20'
'''
#dataset_80_exist_0
'''
window_size = 20
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 377
num_epochs = 349
batch_size = 1024
vocab_size = num_classes + 1
embedding_dim = 400
log = '1_'
kernel = (2,3,4)
event_output_file = 'data/logkey2vec_4.txt'
kernel_dim = 80
train_dataset =  'data/dataset_80_exist_0/train'
model_dir = 'propose_model/dataset_80_exist_0'
'''
#drain_random_with_level_20
'''
window_size = 20
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 1916
num_epochs = 349
batch_size = 512
vocab_size = num_classes + 1
embedding_dim = 400
log = '1_'
kernel = (2,3,4)
event_output_file = 'data/drain_random_with_level_20/event_vector_top4.txt'
kernel_dim = 80
train_dataset =  'data/drain_random_with_level_20/train'
model_dir = 'propose_model/drain_random_with_level_20'
'''
#drain_with_level_80
#window_size = 20
#input_size = 1
#hidden_size = 64
#num_layers = 2
#num_classes = 1916
#num_epochs = 349
#batch_size = 512
#vocab_size = num_classes + 1
#embedding_dim = 400
#log = '1_'
#kernel = (2,3,4)
#event_output_file = 'data/drain_random_with_level_20/event_vector_top4.txt'
#kernel_dim = 80
#train_dataset =  'data/drain_with_level_80/train'
#model_dir = 'propose_model/drain_with_level_80'

#noise_0
'''
window_size = 20
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 1916
num_epochs = 349
batch_size = 128
vocab_size = num_classes + 1
embedding_dim = 400
log = '1_'
kernel = (2,3,4)
event_output_file = 'noise/event_vector_top4.txt'
kernel_dim = 100
train_dataset =  'noise/drain_with_level_80/train'
model_dir = 'noise/our_model_noise_0'
'''
#noise_0 random
# window_size = 20
# input_size = 1
# hidden_size = 64
# num_layers = 2
# num_classes = 1916
# num_epochs = 349
# batch_size = 128
# vocab_size = num_classes + 1
# embedding_dim = 300
# log = '1_'
# kernel = (2,3,4)
# event_output_file = 'noise/event_vector_top2_with_level.txt'
# kernel_dim = 100
# #train_dataset =  'noise/drain_random_with_level_20/train'
# #model_dir = 'noise/our_model_random'
window_size = 20
input_size = 1
#hidden_size = 64
# hidden_size = 128
# num_layers = 2
num_classes = 1916
num_epochs = 60
# batch_size = 128
vocab_size = num_classes + 1
# embedding_dim = 500
# embedding_dim = 400
log = '1_'
#kernel = (2,3,4)
# kernel = (2,3)
# kernel_dim = 80

# event_output_file = 'noise/event_vector_top4_with_level.txt'
# event_output_file = 'noise/event_vector_top4.txt'
#kernel_dim = 100



def generate(name):
    inputs = []
    inputs_q = []
    event2semantic_vec = KeyedVectors.load_word2vec_format(event_output_file, binary=False)

    outputs = []
    with open(name, 'r') as f:
        line_all = ''
        for line in f.readlines():
            line_all += ' '+ line.replace('\n','')

        line_all = tuple(map(int, line_all.strip().split()))

        for i in range(len(line_all) - window_size):
            if i % (window_size+1) == 0:
                Quantitative_pattern = [0] * (num_classes + 1)
                log_counter = Counter(line_all[i:i + window_size])
                Sequential_pattern = line_all[i:i + window_size]
                Semantic_pattern = []

                for event in Sequential_pattern:
                    Semantic_pattern.append(event2semantic_vec[str(event)])

                for key in log_counter:
                    # Quantitative_pattern[key] = log_counter[key]
                    # Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
                    Quantitative_pattern[key] = log_counter[key]
                Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]

                inputs_q.append(Quantitative_pattern)
                inputs.append(Semantic_pattern)
                outputs.append(line_all[i + window_size])

   # print('Number of sessions({}): {}'.format(name, num_sessions))
    #print(inputs[0])
    #print(outputs[0])
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    inputs = torch.tensor(inputs, dtype=torch.float)
   # print(inputs_q[0])
    inputs_q = torch.tensor(inputs_q, dtype=torch.float)
    outputs = torch.tensor(outputs)
    dataset = TensorDataset(inputs, inputs_q,outputs)
    return dataset

'''
class CNNClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim, kernel_sizes=kernel, dropout=0.5):
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

        inputs1 = inputs1.unsqueeze(1)
        inputs1 = [F.relu(conv(inputs1)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        inputs1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs1]  # [(N,Co), ...]*len(Ks)
        concated1 = torch.cat(inputs1, 1)  # torch.Size([2048, 450])
        h0_1 = torch.zeros(num_layers, inputs2.size(0),hidden_size).to(device)
        c0_1 = torch.zeros(num_layers, inputs2.size(0),hidden_size).to(device)
        out1, _ = self.lstm1(inputs2, (h0_1, c0_1))
        multi_out = out1[:, -1, :] # only need lstm head
        concated = torch.cat((concated1, multi_out), 1)
        concated = self.dropout(concated)
        out = self.fc_cnn_lstm(concated)
        return F.log_softmax(out, 1)
'''

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
    train_dataset = sys.argv[1]
    model_dir = sys.argv[2]
    hidden_size = int(sys.argv[3])
    num_layers = int(sys.argv[4])
    kernel = tuple(map(int,sys.argv[5].split(' ')))
    kernel_dim = int(sys.argv[6])
    embedding_dim = int(sys.argv[7])
    event_output_file = sys.argv[8]
    batch_size = int(sys.argv[9])

    model = CNNClassifier(vocab_size, embedding_dim, num_classes, kernel_dim, kernel, 0.5).to(device)
    # model.load_state_dict(torch.load(model_path))

    seq_dataset = generate(train_dataset)
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    writer = SummaryWriter(logdir='log/' + log)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print('window_size:{}'.format(window_size))
    print('num_classes:{}'.format(num_classes))
    print('vocab_size:{}'.format(vocab_size))
    print('num_epochs:{}'.format(num_epochs))
    print('embedding_dim:{}'.format(embedding_dim))
    print('kernel_dim:{}'.format(kernel_dim))
    print('kernel:{}'.format(kernel))
    print('hidden_size = {}'.format(hidden_size))
    print('num_layers = {}'.format(num_layers))
    print('model_dir={}'.format(model_dir))
    print('train_dataset={}'.format(train_dataset))
    print('batch_size={}'.format(batch_size))

    # Train the model
    loss_min = 10
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, q, label) in enumerate(dataloader):
            output = model(seq, q)
            loss = criterion(output, label.to(device))
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], Train_loss: {:.6f}'.format(epoch + 1, num_epochs, train_loss / len(dataloader.dataset)))
        writer.add_scalar('train_loss', train_loss / len(dataloader.dataset), epoch + 1)
        if (train_loss / len(dataloader.dataset) < loss_min) or (epoch > 100):
            loss_min = train_loss / len(dataloader.dataset)
            torch.save(model.state_dict(), model_dir + '/' + log + str(train_loss / len(dataloader.dataset))+'_'+str(epoch))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    writer.close()
    print('Finished Training')
