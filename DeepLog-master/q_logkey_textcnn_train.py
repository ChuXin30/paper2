import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import torch.nn.functional as F
from collections import Counter

# Device configuration
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Hyperparameters
window_size = 10
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 29
#num_classes = 30

num_epochs = 299
batch_size = 2048
vocab_size = num_classes + 1

embedding_dim = 128
model_dir = 'model_q'
kernel_dim = 150

#output_f = 10#log='/0915/test_0915_'#F1-measure: 97.456%
#output_f = 20 #log='/0915_output_20/test_0915_'#F1-measure: 97.481%
#output_f = 30 #log='/0915_output_20/test_0915_'
#output_f = 40 #log='/0915_output_20/test_0915_'

kernel = (2,3,4)
#kernel = (3,4,5)

#kernel = (1,2,3,4)
#log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)+'kernel_dim='+str(kernel_dim)+'embedding_dim=256'
#log='199_best'
#log='/0915/test_0915_'
#log='/0915_output_20/test_0915_'
#log='/0915_output_30/test_0915_'
log='/0915_output_40/test_0915_'

#model_path ='model/test0913'

def generate(name):
    num_sessions = 0
    inputs = []
    inputs_q = []

    outputs = []
    with open('data/' + name, 'r') as f:
    #with open('wordkey_data/' + name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple( map(int, line.strip().split()))
            for i in range(len(line) - window_size):
                Quantitative_pattern = [0] *  (num_classes+1)
                log_counter = Counter(line[i:i + window_size])
                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]

              #  print(line[i:i + window_size])
              #  print(Quantitative_pattern)
                inputs_q.append(Quantitative_pattern)                    
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))

    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float),torch.tensor(inputs_q, dtype=torch.float),torch.tensor(outputs))
    return dataset
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
#    model = CNNClassifier(vocab_size, embedding_dim, num_classes, kernel_dim, (2,3,4), 0.5).to(device)
    model = CNNClassifier(vocab_size, embedding_dim, num_classes, kernel_dim, kernel, 0.5).to(device)

    #model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
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

    # Train the model

    loss_min = 10
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq,q, label) in enumerate(dataloader):
            # Forward pass
            #seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            seq = seq.clone().detach().view(-1, window_size).to(device)
            seq = seq.to(torch.int64) 

            qua = q.clone().detach().view(-1, (num_classes+1)).to(device)
            qua = qua.to(torch.int64) 

            output = model(seq,qua)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], Train_loss: {:.6f}'.format(epoch + 1, num_epochs, train_loss / len(dataloader.dataset)))
        writer.add_scalar('train_loss', train_loss / len(dataloader.dataset), epoch + 1)
        if train_loss / len(dataloader.dataset) < loss_min:
            loss_min = train_loss / len(dataloader.dataset)
            torch.save(model.state_dict(), model_dir  +'/'+ log+str(train_loss / len(dataloader.dataset)))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    writer.close()
    print('Finished Training')
