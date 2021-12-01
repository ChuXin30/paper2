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

# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Hyperparameters
window_size = 10
#window_size = 20

input_size = 1
#hidden_size = 128
hidden_size = 64

num_layers = 2
num_classes = 29
# num_classes = 30

num_epochs = 299
batch_size = 2048
vocab_size = num_classes + 1

#embedding_dim = 100 #top1
#embedding_dim = 300#top3
#embedding_dim = 200#top2
#embedding_dim = 400#top4
embedding_dim = 600#top6

#model_dir = 'new_template2vec/model'#top3
#model_dir = 'new_template2vec/model_window_20'
#model_dir = 'new_template2vec/model_top1'#top1
#model_dir = 'new_template2vec/model_top2'#top2
#model_dir = 'new_template2vec/model_top4'#top4
model_dir = 'new_template2vec/model_only_lstm'#
#model_dir = 'new_template2vec/model_top4_tfidf'#top4

log = '1_'
#kernel_dim = 300
kernel_dim = 150

kernel = (2, 3, 4)
#model_path = 'new_template2vec/model/0.00011261914503299224'
#event_output_file = 'new_template2vec/event_vector_top1.txt'#top1
#event_output_file = 'new_template2vec/event_vector_top2.txt'#top2
event_output_file = 'new_template2vec/event_vector_top4.txt'#top4
#event_output_file = 'new_template2vec/event_vector_top3.txt'
#event_output_file = 'new_template2vec/event_vector_top6.txt'#top6

# = 'new_template2vec/event_vector_top4_tfidf.txt'#top4

def generate(name):
    num_sessions = 0
    inputs = []
    inputs_q = []
    event2semantic_vec = KeyedVectors.load_word2vec_format(event_output_file, binary=False)

    outputs = []
    with open('data/' + name, 'r') as f:
        # with open('wordkey_data/' + name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(int, line.strip().split()))
            for i in range(len(line) - window_size):
                Quantitative_pattern = [0] * (num_classes + 1)
                log_counter = Counter(line[i:i + window_size])
                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]

                #  print(line[i:i + window_size])
                #  print(Quantitative_pattern)
                Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
                #print(Quantitative_pattern.shape)#(30, 1)
                Sequential_pattern = line[i:i + window_size]
                Semantic_pattern = []
                for event in Sequential_pattern:
                    if event == 0:
                        Semantic_pattern.append([-1] * embedding_dim)
                    else:
                        Semantic_pattern.append(event2semantic_vec[str(event -1)])
                #Semantic_pattern = np.array(Semantic_pattern)[:, np.newaxis]
               # print(Semantic_pattern.shape)
                inputs_q.append(Quantitative_pattern)
                inputs.append(Semantic_pattern)
                outputs.append(line[i + window_size])
    #print(inputs[0].shape)
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))

    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(inputs_q, dtype=torch.float),
                            torch.tensor(outputs))
    return dataset


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
        self.fc_cnn_lstm = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs1, inputs2):
        h0_1 = torch.zeros(num_layers, inputs2.size(0),hidden_size).to(device)
        c0_1 = torch.zeros(num_layers, inputs2.size(0),hidden_size).to(device)
        out1, _ = self.lstm1(inputs2, (h0_1, c0_1))
        multi_out = out1[:, -1, :] # only need lstm head

        concated = self.dropout(multi_out)
        out = self.fc_cnn_lstm(concated)
        return F.log_softmax(out, 1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('-num_layers', default=2, type=int)
    #parser.add_argument('-hidden_size', default=64, type=int)
    #parser.add_argument('-window_size', default=10, type=int)
    args = parser.parse_args()

    #window_size = args.window_size
    #    model = CNNClassifier(vocab_size, embedding_dim, num_classes, kernel_dim, (2,3,4), 0.5).to(device)
    model = CNNClassifier(vocab_size, embedding_dim, num_classes, kernel_dim, kernel, 0.5).to(device)
    #model.load_state_dict(torch.load(model_path))

    # model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    seq_dataset = generate('hdfs_train')
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
    # Train the model

    loss_min = 10
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, q, label) in enumerate(dataloader):
            # Forward pass
            # seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
           # seq = seq.clone().detach().view(-1, window_size).to(device)
           # seq = seq.to(torch.int64)

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
