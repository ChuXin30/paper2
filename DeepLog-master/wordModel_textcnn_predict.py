import torch
import torch.nn as nn
import time
import argparse
import torch.nn.functional as F

# Device configuration
device = torch.device("cpu")
# Hyperparameters
input_size = 1

num_classes = 26
num_epochs = 199
batch_size = 2048
vocab_size = num_classes + 1
embedding_dim = 128
model_dir = 'model_word'
kernel_dim = 150
window_size = 10


num_candidates = 4
model_dir = 'model_word'
#log = 'min_lossword_Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)+'kernel_dim='+str(kernel_dim)+'embedding_dim=128'
#model_path = model_dir + '/' + log + '.pt'

kernel = (2,3,4)
log='199_best'
#model_path ='model_word/min_loss199_best.pt'
model_path ='model_word/0914_test1'

def generate(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    #hdfs = set()
    hdfs = []
    with open('wordkey_data/' + name, 'r') as f:
        for line in f.readlines():
            line = list(map(int, line.strip().split()))
         #   line = line + [-1] * (window_size + 1 - len(line))
            line = line + [26] * (window_size + 1 - len(line))

            #hdfs.add(tuple(line))
            hdfs.append(tuple(line))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


class  CNNClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim, kernel_sizes=(2,3,4), dropout=0.5):
        super(CNNClassifier,self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #print(self.embedding)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])
        '''
        上面是个for循环，不好理解写成下面也是没问题的。
        self.conv13 = nn.Conv2d(Ci, Co, (2, D))
        self.conv14 = nn.Conv2d(Ci, Co, (3, D))
        self.conv15 = nn.Conv2d(Ci, Co, (4, D))
        '''
        # kernal_size = (K,D) 
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)
    

    def forward(self, inputs):
      #  print(inputs.shape)
      #  print(inputs)
        inputs = self.embedding(inputs).unsqueeze(1) # (B,1,T,D)
        #inputs = self.embedding(inputs)
       # print(inputs.shape)
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs] #[(N,Co,W), ...]*len(Ks)
        #print(inputs[0].shape)
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs] #[(N,Co), ...]*len(Ks)
        concated = torch.cat(inputs, 1)
        #print(concated.shape)
        #if is_training:
        concated = self.dropout(concated) # (N,len(Ks)*Co)
        out = self.fc(concated) 
        #print(out.shape)
        return F.log_softmax(out,1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    model = CNNClassifier(vocab_size, embedding_dim, num_classes, kernel_dim, (2,3,4), 0.5).to(device)

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



    test_normal_loader = generate('hdfs_test_normal')
    test_abnormal_loader = generate('hdfs_test_abnormal')
    TP = 0
    FP = 0
    count = 0

    # Test the model
    start_time = time.time()




    with torch.no_grad():
        for line in test_abnormal_loader:
            count+=1
            if count %1000 ==0:
                print("count={} TP={}".format(count,TP))
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                #seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size).to(device)
                seq = seq.to(torch.int64) 
                output = model(seq)

                label = torch.tensor(label).view(-1).to(device)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    break
    print("count={} TP={}".format(count,TP))
    count = 0

    with torch.no_grad():
        for line in test_normal_loader:
            count+=1
            if count %1000 ==0:
                print("count={} FP={}".format(count,FP))
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                #seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size).to(device)
                seq = seq.to(torch.int64) 
                output = model(seq)

                label = torch.tensor(label).view(-1).to(device)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1
                    break

    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
