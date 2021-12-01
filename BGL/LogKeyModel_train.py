import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import  sys
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
'''
window_size = 5
input_size = 1
hidden_size = 256
num_layers = 1
#num_classes = 377
num_epochs = 299
batch_size = 2048
model_dir = 'deeplog_model'
#log='deeplog'
#log =  'deeplog2'#使用的是前百分之1的数据集
#log =  'deeplog_dataset10/deeplog_'#使用的是前百分之10的数据集
#log =  'dataset_1/deeplog_'#使用的是前百分之10的数据集

#drain
num_classes = 1850
log =  'drain_struct_dataset_1/deeplog_'
train_dataset = 'struct_data/drain_dataset_1/train'
'''
#dataset 80
'''
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 377
num_epochs = 299
batch_size = 2048
log =  'drain_struct_dataset_1/deeplog_'
#train_dataset = 'data/dataset_80/train'
#model_dir = 'deeplog_model/dataset_80_window_20'
train_dataset = 'data/order_random_dataset_20/train'
model_dir = 'deeplog_model/order_random_dataset_20'
'''
#dataset order_random_dataset_20
'''
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 377
num_epochs = 299
batch_size = 2048
log =  'drain_struct_dataset_1/deeplog_'
train_dataset = 'data/order_random_dataset_20/train'
model_dir = 'deeplog_model/order_random_dataset_20'
'''
#dataset 80 exist 0
'''
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 377
num_epochs = 299
batch_size = 1024
log =  'drain_struct_dataset_1/deeplog_'
train_dataset = 'data/dataset_80_exist_0/train'
model_dir = 'deeplog_model/dataset_80_exist_0'
'''
#dataset 80 exist 0
'''
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 377
num_epochs = 299
batch_size = 1024
log =  'drain_struct_dataset_1/deeplog_'
train_dataset = 'data/order_random_dataset_20_exist_0/train'
model_dir = 'deeplog_model/order_random_dataset_20_exist_0'
'''

#drain_dataset_80
'''
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 1849
num_epochs = 299
batch_size = 1024
log =  'drain_struct_dataset_1/deeplog_'
train_dataset = 'data/drain_dataset_80/train'
model_dir = 'deeplog_model/drain_dataset_80'
'''
#drain_with_level_80

#window_size = 20
#input_size = 1
#hidden_size = 128
#num_layers = 2
#num_classes = 1916
#num_epochs = 299
#batch_size = 1024
#log =  'drain_struct_dataset_1/deeplog_'
#train_dataset = 'data/drain_with_level_80/train'
#model_dir = 'deeplog_model/drain_with_level_80'

#drain_random_with_level_20
'''
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 1916
num_epochs = 299
batch_size = 1024
log =  'drain_struct_dataset_1/deeplog_'
train_dataset = 'data/drain_random_with_level_20/train'
model_dir = 'deeplog_model/drain_random_with_level_20'
'''

#noise_0
'''
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 1916
num_epochs = 299
batch_size = 128
log =  'drain_struct_dataset_1/deeplog_'
train_dataset = 'noise/drain_with_level_80/train'
model_dir = 'noise/deeplog_model_noise_0'
'''

#noise random
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 1916
num_epochs = 299
batch_size = 128
log =  'drain_struct_dataset_1/deeplog_'
#train_dataset = 'noise/drain_random_with_level_20/train'
#model_dir = 'noise/deeplog_model_random'
#train_dataset = 'noise/noise_0/WITH_S_train'
#model_dir = 'noise/noise_0_deeplog'


def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    inputs_train = []
    outputs_train = []

    with open(name, 'r') as f:
        line_all = ''
        for line in f.readlines():
            line_all += ' '+ line.replace('\n','')
        #    num_sessions += 1
        line_all = tuple(map(int, line_all.strip().split()))
        '''
        for i in range(len(line_all) - window_size):
            inputs.append(line_all[i:i + window_size])
            outputs.append(line_all[i + window_size])
        print('Number of seqs({}): {}'.format(name, len(inputs)))
        #print(outputs[0])
        #print(inputs[0])
        '''
        dict_logkey = {}
        for var in line_all:
            if var not in dict_logkey:
                dict_logkey[var] = 1
            else:
                dict_logkey[var] += 1
        #print(dict_logkey)
        print(len(dict_logkey))
#        line_train = line_all[:47000] #取前百分之１作为训练数据
        #line_train = line_all[:240000] #取前百分之5作为训练数据

        line_train = line_all

        count = 0
        length_dataset = len(line_train) - window_size
        for i in range(length_dataset):
            if i % (window_size+1) == 0:
                inputs_train.append(line_train[i:i + window_size])
                outputs_train.append(line_train[i + window_size])
                if count  <= 5:
                    print(line_train[i:i + window_size])
                    print(line_train[i + window_size])
                    count += 1

    #x_test_normal,x_train_normal,y_test,y_train = train_test_split(inputs,outputs,test_size=0.01,random_state=0)
    print(len(inputs_train))
    #print(len(y_train))
    #print(x_train_normal[0])
    #print(y_train[0])

    dict_logkey = {}
    for line in inputs_train:
        for var in line:
            if var not in dict_logkey:
                dict_logkey[var] = 1
            else:
                dict_logkey[var] += 1
    #print(dict_logkey)
    print(len(dict_logkey))
    print('Number of seqs({}): {}'.format(name, len(inputs_train)))
    dataset = TensorDataset(torch.tensor(inputs_train, dtype=torch.float), torch.tensor(outputs_train))
    return dataset

def preposess_time_slide(name):
    #f = open('data/time_slide/'+name,mode='r')
    #print(name)
    f = open('data/logkey/'+name,mode='r')    
    list_a = ()
    #line_all = tuple(map(int, line_all.strip().split()))
    inputs = []
    outputs = []
    lable = []
    inputs_q = []
    
    line_all =""
    for line in f:
        if '0' == line.split(' ')[0]:
            lable.append(0)
        else:
            lable.append(1)
        line_all += ' ' + line.split(' ')[1]
    
    list_a = list(map(int, line_all.strip().split()))
    #list_a = [1,2,3,4,5,6]
  #  print(list_a)

    if len(list_a) > window_size: #补齐长度
        list_a = list_a[:window_size] + list_a
    else:
        list_a = [list_a[0]] * (window_size ) +list_a  
  
    list_a = tuple(list_a)
    for i in range(len(list_a) - window_size):
        inputs.append(list_a[i:i + window_size])
        outputs.append(list_a[i + window_size])
        Quantitative_pattern = [0] *  (num_classes+1)
        log_counter = Counter(list_a[i:i + window_size])
        for key in log_counter:
            Quantitative_pattern[key] = log_counter[key]

        inputs_q.append(Quantitative_pattern)     

    return inputs,inputs_q ,outputs,lable

def get_log_time(line):
    time = line.split()[4][:18]
    return str(time)

def generate_1():
    inputs_seq = []
    outputs = []
    inputs_q = []
    lable = []
    count = 0
    time_dict = {}
    f = open('data/BGL.log',mode='r')

    for line in f:
        count += 1
        time = get_log_time(line)
        if time not in time_dict:
            time_dict[time] = time

    count = []
    for var in time_dict:
        s , q, o , l = preposess_time_slide(var)
        for i in range(len(l)):
            if l[i] == 0:#只要正常的数据进行训练
                inputs_seq.append(s[i])
                inputs_q.append(q[i])
                outputs.append(o[i])
                count.append(s[i])
            else:
                count.append(s[i])


    print(len(count))
    dict_logkey = {}
    for line in count:
        for var in line:
            if var not in dict_logkey:
                dict_logkey[var] = 1
            else:
                dict_logkey[var] += 1
    print(len(dict_logkey))

    inputs_seq = inputs_seq[2000000:]
    outputs = outputs[2000000:]

    dict_logkey = {}
    for line in inputs_seq:
        for var in line:
            if var not in dict_logkey:
                dict_logkey[var] = 1
            else:
                dict_logkey[var] += 1
    print(len(dict_logkey))

    x_test_normal,x_train_normal,y_test,y_train = train_test_split(inputs_seq,outputs,test_size=0.002,random_state=0)
    dataset = TensorDataset(torch.tensor(x_train_normal, dtype=torch.float),torch.tensor(y_train))
    print(len(x_train_normal))

    dict_logkey = {}    
    for line in x_train_normal:
        for var in line:
            if var not in dict_logkey:
                dict_logkey[var] = 1
            else:
                dict_logkey[var] += 1
    print(len(dict_logkey))
  #  dataset = TensorDataset(torch.tensor(inputs_seq, dtype=torch.float),torch.tensor(inputs_q, dtype=torch.float),torch.tensor(outputs))
  #  print(dataset[0])
    return dataset

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
    train_dataset = sys.argv[1]
    model_dir = sys.argv[2]
    print('window_size:{}'.format(window_size))
    print('num_classes:{}'.format(num_classes))
    print('input_size:{}'.format(input_size))
    print('num_epochs:{}'.format(num_epochs))
    print('batch_size:{}'.format(batch_size))
    print('log:{}'.format(log))
    print('hidden_size = {}'.format(hidden_size))
    print('num_layers = {}'.format(num_layers))
    print('model_dir={}'.format(model_dir))

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    seq_dataset = generate(train_dataset)
    #seq_dataset = generate_1()
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    writer = SummaryWriter(logdir='log/' + log)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            #seq = seq.to(torch.int64) 

            output = model(seq)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], Train_loss: {:.8f}'.format(epoch + 1, num_epochs, train_loss / len(dataloader.dataset)))
        writer.add_scalar('train_loss', train_loss / len(dataloader.dataset), epoch + 1)
        torch.save(model.state_dict(), model_dir + '/'+  str(train_loss / len(dataloader.dataset)) + '_' + str(epoch))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + '.pt')
    writer.close()
    print('Finished Training')
