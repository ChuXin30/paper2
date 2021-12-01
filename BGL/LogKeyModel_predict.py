import torch
import torch.nn as nn
import time
import argparse
from sklearn.model_selection import train_test_split
import time
import  sys
# Device configuration
device = torch.device("cpu")
# Hyperparameters


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
'''
window_size = 5
input_size = 1
hidden_size = 256
num_layers = 1
num_classes = 377
num_epochs = 299
batch_size = 2048
num_candidates = 40
model_dir = 'deeplog_model'
log =  'deeplog_dataset1/deeplog_'
#model_path = 'deeplog_model/deeplog_dataset10/deeplog_6.472419281868689e-05_108'# 76.694%
#model_path = 'deeplog_model/deeplog_dataset10/deeplog_6.252733708461428e-05_153'
#model_path = 'deeplog_model/deeplog_dataset10/deeplog_7.580765587412319e-05_35'
model_path = 'deeplog_model/dataset_1/deeplog_2.7650249170749193e-05_161'
#normal_test = 'data/dataset_1/test_deeplog'
#abnormal_test='data/dataset_1/abnormal_deeplog'
normal_test = 'data/dataset_1/test_new'
abnormal_test='data/dataset_1/abnormal_new'
#normal_test = 'data/dataset_10/test_new' #F1-measure: 76.962%
#abnormal_test='data/dataset_10/abnormal_new'
'''

#drain
'''
window_size = 5
input_size = 1
hidden_size = 256
num_layers = 1
#num_classes = 377
num_epochs = 299
batch_size = 2048
num_classes = 1850
num_candidates = 400
model_dir = 'deeplog_model'
log =  'drain_struct_dataset_1/deeplog_'
#model_path = 'deeplog_model/drain_struct_dataset_1/deeplog_3.095967839713237e-05_175'
model_path = 'deeplog_model/dataset_1/deeplog_2.4159527338333452e-05_220'
#normal_test = 'struct_data/drain_dataset_1/dataset_test_normal_deeplog'
#abnormal_test='struct_data/drain_dataset_1/dataset_test_abnormal_deeplog'
#normal_test = 'struct_data/drain_dataset_1/dataset_test_normal'
#abnormal_test='struct_data/drain_dataset_1/dataset_test_abnormal'
test_normal_dataset =  'data/dataset_1/test_new'
test_abnormal_dataset =  'data/dataset_1/abnormal_new'
'''
#dataset_80 window 20
'''
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 377
num_epochs = 299
batch_size = 2048
num_candidates = 40
model_dir = 'deeplog_model'
log =  'deeplog_dataset1/deeplog_'
'''

#bgl deeplog predict1
'''
#normal_test = 'data/dataset_80/test_normal'
#abnormal_test='data/dataset_80/test_abnormal'
#model_path = 'deeplog_model/dataset_80_window_20/0.0001486994094951056_37'#false positive (FP): 17455, false negative (FN): 5021, Precision: 41.536%, Recall: 71.180%, F1-measure: 52.460%
#model_path = 'deeplog_model/dataset_80_window_20/8.360735038177901e-05_189'#false positive (FP): 17310, false negative (FN): 5027, Precision: 41.727%, Recall: 71.146%, F1-measure: 52.603%
#num_candidates = 170
#model_path = 'deeplog_model/dataset_80_window_20/8.360735038177901e-05_189'#false positive (FP): 9710, false negative (FN): 5952, Precision: 54.155%, Recall: 65.836%, F1-measure: 59.427%
#num_candidates = 200
#model_path = 'deeplog_model/dataset_80_window_20/8.360735038177901e-05_189'#false positive (FP): 3473, false negative (FN): 5971, Precision: 76.729%, Recall: 65.727%, F1-measure: 70.803%
#num_candidates = 250
#model_path = 'deeplog_model/dataset_80_window_20/8.360735038177901e-05_189'#false positive (FP): 2176, false negative (FN): 6443, Precision: 83.459%, Recall: 63.018%, F1-measure: 71.812%
#num_candidates = 300
#model_path = 'deeplog_model/dataset_80_window_20/8.360735038177901e-05_189'#false positive (FP): 105, false negative (FN): 6591, Precision: 99.040%, Recall: 62.169%, F1-measure: 76.388%
#num_candidates = 350
#model_path = 'deeplog_model/dataset_80_window_20/8.360735038177901e-05_189'#false positive (FP): 28, false negative (FN): 17416, Precision: 17.647%, Recall: 0.034%, F1-measure: 0.069%
#normal_test = 'data/dataset_80/test_normal_new'
#abnormal_test='data/dataset_80/test_abnormal_new'
#model_path = 'deeplog_model/dataset_80_window_20/0.0001486994094951056_37'#false positive (FP): 684, false negative (FN): 14161, Precision: 82.662%, Recall: 18.718%, F1-measure: 30.524%
#num_candidates = 1
#model_path = 'deeplog_model/dataset_80_window_20/8.360735038177901e-05_189'
#normal_test = 'data/order_random_dataset_20/test_normal'
#abnormal_test='data/order_random_dataset_20/test_abnormal'
#model_path = 'deeplog_model/order_random_dataset_20/0.0002465617672449563_179'
#num_candidates = 200#false positive (FP): 132, false negative (FN): 6446, Precision: 98.898%, Recall: 64.762%, F1-measure: 78.270%
#num_candidates = 100#false positive (FP): 254, false negative (FN): 5268, Precision: 98.087%, Recall: 71.202%, F1-measure: 82.510%
#num_candidates = 50#false positive (FP): 883, false negative (FN): 5266, Precision: 93.652%, Recall: 71.213%, F1-measure: 80.906%
#num_candidates = 25#false positive (FP): 2096, false negative (FN): 5255, Precision: 86.150%, Recall: 71.273%, F1-measure: 78.009%
'''

#dataset_80_exist_0
'''
model_path = 'deeplog_model/dataset_80_exist_0/0.00017170341254692204_120'
normal_test = 'data/dataset_80_exist_0/test_normal_new'
abnormal_test='data/dataset_80_exist_0/test_abnormal_new'
#num_candidates = 100#count_ab=1000 TP=892 count=1000 FP=0#false positive (FP): 17257, false negative (FN): 5027, Precision: 41.802%, Recall: 71.146%, F1-measure: 52.662%
#num_candidates = 200#count_ab=1000 TP=892 count=1000 FP=0#false positive (FP): 17105, false negative (FN): 5102, Precision: 41.869%, Recall: 70.715%, F1-measure: 52.597%
num_candidates = 300#count_ab=1000 TP=891 count=1000 FP=0
#num_candidates = 50#count_ab=1000 TP=892 count=1000 FP=0false positive (FP): 17539, false negative (FN): 5026, Precision: 41.410%, Recall: 71.151%, F1-measure: 52.351%
#num_candidates = 30#count_ab=1000 TP=892 count=1000 FP=0
#num_candidates = 10#count_ab=1000 TP=892 count=1000 FP=0 false positive (FP): 18016, false negative (FN): 5017, Precision: 40.778%, Recall: 71.203%, F1-measure: 51.857%
#num_candidates = 5#count_ab=1000 TP=892 count=1000 FP=1
#num_candidates = 1#count_ab=1000 TP=892 count=1000 FP=60
'''
#order_random_dataset_20_exist_0
'''
model_path = 'deeplog_model/order_random_dataset_20_exist_0/0.0005093534634853644_99'
normal_test = 'data/order_random_dataset_20_exist_0/test_normal_new'
abnormal_test='data/order_random_dataset_20_exist_0/test_abnormal_new'
#num_candidates = 350 #count=1000 FP=0
#num_candidates = 300 #count=1000 FP=0
#num_candidates = 200 #count=1000 FP=0 count_ab=1000 TP=96
num_candidates = 100 #count=1000 FP=2 count_ab=1000 TP=707
#num_candidates = 50 #count=1000 FP=5 count_ab=1000 TP=717
#num_candidates = 30 #count=1000 FP=9 count_ab=1000 TP=717
#num_candidates = 10 #count=1000 FP=21 count_ab=1000 TP=721
#num_candidates = 5 #count=1000 FP=34 count_ab=1000 TP=721
#num_candidates = 1 #count=1000 FP=139 count_ab=1000 TP=722
normal_test = 'data/order_random_dataset_20_exist_0/test_normal'
abnormal_test='data/order_random_dataset_20_exist_0/test_abnormal'
num_candidates = 100#false positive (FP): 213, false negative (FN): 5294, Precision: 98.388%, Recall: 71.060%, F1-measure: 82.520%
num_candidates = 150#
'''

#drain_dataset_80
'''
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 1849
#normal_test = 'data/order_random_dataset_20_exist_0/test_normal'
#abnormal_test='data/order_random_dataset_20_exist_0/test_abnormal'
normal_test = 'data/drain_dataset_80/test_normal_simple'
abnormal_test='data/drain_dataset_80/test_abnormal_simple'
model_path = 'deeplog_model/drain_dataset_80/0.00025200240010224564_50'
#num_candidates = 1800#false positive (FP): 5, false negative (FN): 913, Precision: 28.571%, Recall: 0.219%, F1-measure: 0.434%
#num_candidates = 1500#false positive (FP): 514, false negative (FN): 491, Precision: 45.203%, Recall: 46.339%, F1-measure: 45.764%
#num_candidates = 1000#false positive (FP): 632, false negative (FN): 331, Precision: 48.026%, Recall: 63.825%, F1-measure: 54.810%
#num_candidates = 800#false positive (FP): 740, false negative (FN): 284, Precision: 46.025%, Recall: 68.962%, F1-measure: 55.206%
#num_candidates = 700#false positive (FP): 943, false negative (FN): 255, Precision: 41.173%, Recall: 72.131%, F1-measure: 52.423%
#num_candidates = 600#false positive (FP): 974, false negative (FN): 232, Precision: 41.219%, Recall: 74.645%, F1-measure: 53.110%
#num_candidates = 500#false positive (FP): 994, false negative (FN): 203, Precision: 41.735%, Recall: 77.814%, F1-measure: 54.330
#num_candidates = 400#false positive (FP): 998, false negative (FN): 156, Precision: 43.199%, Recall: 82.951%, F1-measure: 56.811%
#num_candidates = 350#false positive (FP): 1003, false negative (FN): 93, Precision: 45.041%, Recall: 89.836%, F1-measure: 60.000%
#num_candidates = 250#false positive (FP): 1008, false negative (FN): 12, Precision: 47.253%, Recall: 98.689%, F1-measure: 63.907%
#num_candidates = 200#false positive (FP): 1010, false negative (FN): 12, Precision: 47.203%, Recall: 98.689%, F1-measure: 63.861%
#num_candidates = 150#false positive (FP): 1012, false negative (FN): 12, Precision: 47.154%, Recall: 98.689%, F1-measure: 63.816%
#num_candidates = 100#false positive (FP): 1016, false negative (FN): 12, Precision: 47.056%, Recall: 98.689%, F1-measure: 63.726%
#num_candidates = 50#false positive (FP): 1022, false negative (FN): 9, Precision: 46.992%, Recall: 99.016%, F1-measure: 63.735%

'''
#drain_with_level_80
'''
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 1916
#normal_test = 'data/order_random_dataset_20_exist_0/test_normal'
#abnormal_test='data/order_random_dataset_20_exist_0/test_abnormal'
#normal_test = 'data/drain_with_level_80/test_normal_simple'
#abnormal_test='data/drain_with_level_80/test_abnormal_simple'
normal_test = 'data/drain_with_level_80/test_normal'
abnormal_test='data/drain_with_level_80/test_abnormal'
#model_path = 'deeplog_model/drain_with_level_80/0.00027601338558450234_46'#
model_path = 'deeplog_model/drain_with_level_80/0.0003237142248483717_33'#
#num_candidates = 500#false positive (FP): 945, false negative (FN): 16, Precision: 46.276%, Recall: 98.072%, F1-measure: 62.881%
#num_candidates = 500#
#num_candidates = 500#
#num_candidates = 500#
#num_candidates = 500#
#num_candidates = 800#false positive (FP): 935, false negative (FN): 504, Precision: 25.852%, Recall: 39.277%, F1-measure: 31.181%
'''
#drain_random_with_level_20
'''
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 1916
#normal_test = 'data/drain_random_with_level_20/test_normal_simple'
#abnormal_test='data/drain_random_with_level_20/test_abnormal_simple'
normal_test = 'data/drain_random_with_level_20/test_normal'
abnormal_test='data/drain_random_with_level_20/test_abnormal'
model_path = 'deeplog_model/drain_random_with_level_20/0.0004171226096161388_148'#
#num_candidates = 500#false positive (FP): 13, false negative (FN): 73, Precision: 98.399%, Recall: 91.628%, F1-measure: 94.893%
'''
#drain_random_with_level_20
window_size = 20
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 1916
#normal_test = 'noise/drain_random_with_level_20/test_normal_simple'
#abnormal_test='noise/drain_random_with_level_20/test_abnormal_simple'
model_path = 'noise/deeplog_model_random/0.003090536963236523_129'#
#model_path = 'noise/noise_0_deeplog/0.004234337004164491_176'#

def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    with open( name, 'r') as f:
        line_all = ''
        for line in f.readlines():
            line_all += ' '+ line.replace('\n','')
        #    num_sessions += 1
        line_all = tuple(map(int, line_all.strip().split()))
        for i in range(len(line_all) - window_size):
            if i%(window_size+1)  ==0:
                inputs.append(line_all[i:i + window_size])
                outputs.append(line_all[i + window_size])
        print('Number of seqs({}): {}'.format(name, len(inputs)))
    return inputs,outputs

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

    num_candidates = int(sys.argv[1])
    normal_test = sys.argv[2]
    abnormal_test = sys.argv[3]

    print('window_size:{}'.format(window_size))
    print('num_classes:{}'.format(num_classes))
    print('input_size:{}'.format(input_size))
    print('hidden_size = {}'.format(hidden_size))
    print('num_layers = {}'.format(num_layers))
    print('normal_test = {}'.format(normal_test))
    print('abnormal_test={}'.format(abnormal_test))
    print('num_candidates={}'.format(num_candidates))
    print('model_path={}'.format(model_path))


    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    test_normal_seq , test_normal_lable = generate(normal_test)
    test_abnormal_seq, test_abnormal_lable  = generate(abnormal_test)
    TP = 0
    FP = 0
    count = 0
    # Test the model
    start_time = time.time()

    count_ab = 0
    for var in range(len(test_abnormal_seq)):
        count_ab+=1
        if count_ab %500 ==0:
            print("count_ab={} TP={}".format(count_ab,TP))
        label = test_abnormal_lable[var]
        seq = test_abnormal_seq[var]
        seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
        label = torch.tensor(label).view(-1).to(device)
        output = model(seq)
        predicted = torch.argsort(output, 1)[0][-num_candidates:]
        if label not in predicted:
            TP += 1
            continue

    count = 0
    for var in range(len(test_normal_seq)):
        count+=1
        if count %1000 ==0:
            print("count={} FP={}".format(count,FP))
        label = test_normal_lable[var]
        seq = test_normal_seq[var]
        seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
        label = torch.tensor(label).view(-1).to(device)
        output = model(seq)
        predicted = torch.argsort(output, 1)[0][-num_candidates:]
        if label not in predicted:
            FP += 1
            continue

    #print(len(test_abnormal_seq))


    # Compute precision, recall and F1-measure
    FN = count_ab - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
