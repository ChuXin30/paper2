# -*- coding: utf-8 -*
import pandas as pd
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import re
from sklearn.model_selection import train_test_split
import numpy as np
#from tflearn.data_utils import to_categorical, pad_sequences
import  os
import gensim
import numpy as np
from gensim.models import word2vec
 # -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time
from LogKeyModel_predict import Model
from bgl_train import CNNClassifier
from collections import Counter

def get_blk_id(line):
    # 'blk_-1608999687919862906'
    #s = '081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906\n src: /10.250.19.102:54106 dest: /10.250.19.102:50010\n'
    obj = re.search(r" blk_([0-9]*?)\. (.*?)", line)
    blk_id = '1'
    if obj != None:
     #   print('1')
        #print(obj.group(1))
        blk_id = obj.group(1)
    else:
        obj2 = re.search(r" blk_([0-9]*?)\n", line)
        if obj2 != None:
      #      print('2')
            #print(obj.group(1))
            blk_id = obj2.group(1)
        else:
            obj3 = re.search(r" blk_([0-9]*?) (.*?)", line)
            if obj3 != None:
       #         print('99999')
                blk_id = obj3.group(1)

    if blk_id != '1':
        blk_id = 'blk_'+blk_id
        return blk_id,line


    obj = re.search(r" blk_-([0-9]*?)\. (.*?)", line)
    blk_id = '1'
    if obj != None:
     #   print('1')
        #print(obj.group(1))
        blk_id = obj.group(1)
    else:
        obj2 = re.search(r" blk_-([0-9]*?)\n", line)
        if obj2 != None:
      #      print('2')
            #print(obj.group(1))
            blk_id = obj2.group(1)
        else:
            obj3 = re.search(r" blk_-([0-9]*?) (.*?)", line)
            if obj3 != None:
       #         print('99999')
                blk_id = obj3.group(1)

    if blk_id != '1':
        blk_id = 'blk_-'+blk_id

   # print(blk_id)
    return blk_id ,line



def sub_head(line):#去除头部时间戳、日志ｉｄ　只保留日志的信息
    line_without_head = ''
    line_list = line.split(" ")[5:]
    line_without_head = ' '.join(line_list)
    return line_without_head

def group_blk_id():
    print(word2vec)
    #f = open('data/HDFS.log',mode='r')
    #csv_lable = pd.read_csv('data/anomaly_label.csv',nrows =num)
    csv_lable = pd.read_csv('data/anomaly_label.csv')
    print(csv_lable)
    csv_lable = csv_lable.values.tolist()
    
    dict_blk_id = {}
    dict_blk_id_label = {}
    for csv_lable_line in csv_lable:
        dict_blk_id[csv_lable_line[0]]  = ''
        dict_blk_id_label[csv_lable_line[0]] = csv_lable_line[1]

    f_dict_word = open('word_in_clo_header.txt',mode='r')
    dict_word = {}
    for var in f_dict_word:
        var =var.replace('\n','').lower()
        dict_word[var] = 0
    print(dict_word)
    print(len(dict_word))

    f = open('data/HDFS.log', mode='r') #将数据按照blk_id进行group
    j= 0
    for line in f.readlines():
        if j %100000 == 0:
            print(j)
        blk_id,s = get_blk_id(line)
        line = sub_head(line)
        #print(blk_id)
        if blk_id in dict_blk_id:
            dict_blk_id[blk_id] = dict_blk_id[blk_id] + line
        else:
            print(blk_id)
            print(s)
        j = j+1
   # print(dict_blk_id['blk_-1608999687919862906'])
#
    print(dict_blk_id['blk_-1067131609371010449'])
    print(dict_blk_id['blk_-1608999687919862906'])    
#    r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
#    r1 = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    j = 0
    list_normal = []
    list_normal_lable= []

    f_dict_abnormal = open('data_word/hdfs_test_abnormal',mode='w')

    for key in dict_blk_id:
        line_blk_vac = ''
#        dict_blk_id[key] = sub_num_ip_filename(dict_blk_id[key],key)# 去除ｉｐ地址和 blk_id
#        dict_blk_id[key] = re.sub(r1, ' ', dict_blk_id[key])
        r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        dict_blk_id[key] = re.sub(r1, ' ', dict_blk_id[key])       
        dict_blk_id[key] = dict_blk_id[key].lower()  #blk_-1067131609371010449
        dict_blk_id[key].replace('\n',' ')
        for var in dict_blk_id[key].split(' '):
            if var in dict_word:
                line_blk_vac += ' ' + var
        if dict_blk_id_label[key] == 'Normal':
            list_normal.append(line_blk_vac)
            list_normal_lable.append(0)
        else:
            f_dict_abnormal.write(line_blk_vac+'\n')

        if j %10000 == 0:
            print(j)
        j = j +1
    f_dict_test_normal = open('data_word/hdfs_test_normal',mode='w')
    f_dict_train_normal = open('data_word/hdfs_train',mode='w')

    x_test_normal,x_train_normal,y_train,y_test = train_test_split(list_normal,list_normal_lable,test_size=0.01,random_state=0)


    for line in x_test_normal:
        f_dict_test_normal.write(line+'\n')
    for line in x_train_normal:
        f_dict_train_normal.write(line+'\n')

    f_dict_test_normal.close()
    f_dict_train_normal.close()

def create_word_in_col_head():
    f = open('col_header.txt',mode='r')
    fw = open('word_in_clo_header.txt',mode='w')
    dict_word =  {}
    for line in f:
        r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = line.replace('\n',' ').lower()
        line = re.sub(r1, ' ' , line)
        for var in line.split(' '):
            if var not in dict_word:
                dict_word[var] = 1
                fw.write(var+'\n')
    print(dict_word)
    print(len(dict_word))

def word_to_key(name):
    f_dict_word = open('word_in_clo_header.txt',mode='r')
    dict_word = {}
    count = 0
    for var in f_dict_word:
        var =var.replace('\n','').lower()
        dict_word[var] = count 
        count += 1
    print(dict_word)
    print(len(dict_word))

    fr = open('data_word/'+name,mode='r')
    fw = open('wordkey_data/'+name,mode='w')
    for line in fr:
        line_var = ''
        for var in line.split(' '):
            if var in dict_word:
                line_var += ' ' + str(dict_word[var]) 
        fw.write(line_var.strip()+'\n')

def cout_normal():
    f = open('data/BGL.log',mode='r')
    '''
    normal_d = {}
    abnormal_d = {}
    first_word = {}
    for line in f:
        first = line.split(' ')[0]
        if first not in first_word:
            first_word[first] = 1
        else:
            first_word[first] += 1
        if '- ' not in line:
            #line.split()[]
            var = line.split(' ')[1]
            if var not in abnormal_d: 
                abnormal_d[var] = 1
            else:
                abnormal_d[var] += 1
        else:
            var = line.split(' ')[1]
            if var not in normal_d: 
                normal_d[var] = 1
            else:
                normal_d[var] += 1 
    print(len(abnormal_d))           
    #print(normal_d)
    print(len(normal_d)) 
    print(first_word)
    print(len(first_word))
    '''


    normal = {}
    all_d= {}
    for line in f:
        first = line.strip().split(' ')[0]
        blg_id =  line.strip().split(' ')[1]
      #  print("first={} blg_id={}".format(first,blg_id))
        if blg_id not in all_d:
            all_d[blg_id] = 0
        else:
            all_d[blg_id] += 1

        if first == '-':
            if blg_id not in normal:
                normal[blg_id] = 0
            else:
                normal[blg_id] += 1

    sub_normal = {}
    for var in all_d:
        if var not in normal:
            sub_normal[var] = 0
    print(sub_normal)
    print(len(all_d))
    print(len(normal))
    print(len(sub_normal))





window_size = 5    
def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    with open('data/' + name, 'r') as f:
        line_all = ''
        for line in f.readlines():
            line_all += ' '+ line.replace('\n','')

        #    num_sessions += 1
        line_all = tuple(map(int, line_all.strip().split()))
        for i in range(len(line_all) - window_size):
            inputs.append(line_all[i:i + window_size])
            outputs.append(line_all[i + window_size])
        print('Number of seqs({}): {}'.format(name, len(inputs)))
       # print(outputs[0])
       # print(inputs[0])
        dict_logkey = {}
        for var in line_all:
            if var not in dict_logkey:
                dict_logkey[var] = 1
            else:
                dict_logkey[var] += 1
      #  print(dict_logkey)
        print(len(dict_logkey))
    x_test_normal,x_train_normal,y_train,y_test = train_test_split(inputs,outputs,test_size=0.01,random_state=0)
    print(len(x_train_normal))
   # print(x_train_normal[0])
   # print(y_train[0])

    dict_logkey = {}
    for line in x_train_normal:
        for var in line:
            if var not in dict_logkey:
                dict_logkey[var] = 1
            else:
                dict_logkey[var] += 1
    #print(dict_logkey)
    print(len(dict_logkey))
    return inputs,outputs
   # print(len(x_test_normal))
    #dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    #return dataset

def get_log_time(line):
    time = line.split()[4][:18]
    return str(time)


match_temp = {}
f = open('data/BGL_templates.csv',mode='r')
fw = open("data/templates",mode='w')
for line in f:
    num = line.split(',')[0][1:]
    print(line)
    #key = re.escape(line.split(',')[1].replace('\n','')).replace('<\\*>','(.*)').replace("\\",'')
    key = line.split(',')[1].replace('(','\(').replace(')','\)').replace('<*>','(.*)').replace("\"\"",'').replace('\n','')
    key = key.replace('=','\=').replace('|','\|').replace("\"","").strip()
    key = key.replace('[','\[').replace(']','\]').replace('$','\$').replace('/','\/')
    key = key.replace('-','\-')
    if key not in match_temp:
        match_temp[num] = key
        fw.write(num+','+key+'\n')
#print(match_temp)
#print(len(match_temp))

def match_log_key(s):
   # print(s)
    s = s.replace('\n','')
 #   print(s)

    for var in match_temp:
       # print(var)
        obj = re.search( match_temp[var],s)
        if obj!= None:
            return var
    print(s)
    return '0'



def split_log_by_time():
    f = open('data/BGL.log',mode='r')
    time_dict = {}
    count = 0
    for line in f:
        count += 1
        #print(get_log_time(line))
        time = get_log_time(line)
        if time not in time_dict:
            time_dict[time] = line
        else:
            time_dict[time] = time_dict[time] +'\n' + line
    for var in time_dict:
        #print(time_dict[var])
        f = open('data/time_slide/'+var,mode='w')
        for line in time_dict[var].split('\n'):
           if line != '':
               f.write(line+'\n')
           # line = line+'\n'
           # f.write(line.replace('\n\n','\n'))
#    print(time_dict)
#    print(len(time_dict)) #61887
#    print(count) #4747963
    count = 1
    for var in time_dict:
        f = open('data/time_slide/'+var,mode='r')
        for line in f:
            count += 1

    print(count)#4747964




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


device = torch.device("cpu")
input_size = 1
hidden_size = 64
num_layers = 2
num_candidates = 20
num_classes = 377
num_epochs = 199
batch_size = 2048
vocab_size = num_classes + 1
window_size = 5
embedding_dim = 128
model_dir = 'propose_model/model'
kernel_dim = 150
kernel = (2,3,4)
#model_path ='model/test1_best' #使用百分之１的数据集训练
#model_path ='model/text_cnn50' #使用前百分之５０的数据集训练
#model_path ='q_model/seq/0916_7.400718675178661e-05' 
#model_path ='q_model/seq_key100/0916_test2_4.1133277490735055e-05' #F1-measure: 95.025%

model_path ='q_model/text_cnn_key120/0916_test2_9.997583304842313e-05' #F1-measure: 95.025%


model = CNNClassifier(vocab_size, embedding_dim, num_classes, kernel_dim, kernel, 0.5).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

def predicted(inputs,outputs,flag):
    TP = 0
    FP = 0    
    for var in range(len(inputs)):
        label = outputs[var]
        seq = inputs[var]
        #seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
        seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size).to(device)
        seq = seq.to(torch.int64) 
        output = model(seq)

        label = torch.tensor(label).view(-1).to(device)
        predicted = torch.argsort(output, 1)[0][-num_candidates:]

        if flag[var] == 0: 
            if label  in predicted:
                TP += 1        
        else:
            if label  in predicted:
                FP += 1
    return TP,FP


window_size = 5
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 377
num_epochs = 100
model_path = 'q_model/deeplog_key120/deeplog.pt'
model_deeplog = Model(input_size, hidden_size, num_layers, num_classes).to(device)
model_deeplog.load_state_dict(torch.load(model_path))
model_deeplog.eval()

def predict_deeplog(inputs,outputs,flag):
    num_candidates = 20
    TP = 0
    FP = 0    
    for var in range(len(inputs)):
        label = outputs[var]
        seq = inputs[var]
        #seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
        seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
        label = torch.tensor(label).view(-1).to(device)
        output = model_deeplog(seq)
        predicted = torch.argsort(output, 1)[0][-num_candidates:]

        if flag[var] == 0: 
            if label  in predicted:
                TP += 1        
        else:
            if label  in predicted:
                FP += 1
    return TP,FP    

def predict_all():
    time_dict = {}
    count = 0
    f = open('data/BGL.log',mode='r')

    for line in f:
        count += 1
        time = get_log_time(line)
        if time not in time_dict:
            time_dict[time] = time
    count_dataset = 0
    TP_total = 0
    FP_total = 0
    for var in time_dict:
        inputs,q,outputs,flag = preposess_time_slide(var)
        count_dataset += len(inputs)
        #TP ,FP =  predicted(inputs,outputs,flag) #text_cnn
        TP ,FP =  predict_deeplog(inputs,outputs,flag) #deeplog
        TP_total += TP
        FP_total += FP

        FN = count_dataset - TP_total
        P = 100 * TP_total*1.0 / (TP_total + FP_total+1)
        R = 100 * TP_total*1.0 / (TP_total + FN+1)
        if (P+R) != 0:
            F1 = 2.0 * P * R / (P + R)
        else:
            F1 = 0
        #print("".format( ))

        print('count_dataset={} TP_total={} FP_total={}  false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(count_dataset ,TP_total,FP_total, FN, P, R, F1))


def logfile_to_logkeyfile():
    time_dict = {}
    count = 0
    count_dataset = 0
    f = open('data/BGL.log',mode='r')

    for line in f:
        count += 1
        time = get_log_time(line)
        if time not in time_dict:
            time_dict[time] = time

    for var in time_dict:
        inputs,outputs,flag = preposess_time_slide(var)
        count_dataset += len(inputs)

        f = open('data/logkey/'+var,mode='w')
        for i in range(len(outputs)):
            f.write(str(str(flag[i])+' '+str(outputs[i]))+'\n')
        print('count_dataset={} '.format(count_dataset))

def count_logkey_in_train():

    dict_logkey = {}    
    with open('data/' + 'normal', 'r') as f:
        line_all = ''
        for line in f.readlines():
            line_all += ' '+ line.replace('\n','')
        line_all = tuple(map(int, line_all.strip().split()))
        for var in line_all:
            if var not in dict_logkey:
                dict_logkey[var] = 1
            else:
                dict_logkey[var] += 1

    with open('data/' + 'abnormal', 'r') as f:
        line_all = ''
        for line in f.readlines():
            line_all += ' '+ line.replace('\n','')
        line_all = tuple(map(int, line_all.strip().split()))
        for var in line_all:
            if var not in dict_logkey:
                dict_logkey[var] = 1
            else:
                dict_logkey[var] += 1
    print(len(dict_logkey))

def get_importain_word(name,imp):
    fr = open(name,mode='r')
    corpus = []
    for line in fr:
        line = line.replace('\n',' ')
        line2 = ''
        for var in line.split(' '):
            if '.' not in var and '\\' not in var:
                line2 += ' '+var
        corpus.append(line2)
    #将文本中的词语转换为词频矩阵  

    vectorizer = CountVectorizer(lowercase=False)  
    X = vectorizer.fit_transform(corpus)  
    word = vectorizer.get_feature_names()     
    print(word)  

    transformer = TfidfTransformer()  
   # print(transformer)  
    #将词频矩阵X统计成TF-IDF值  
    tfidf = transformer.fit_transform(X)  
    weight = tfidf.toarray()
    print(len(word))

    importain_word = {}
    for i in range(len(weight)):
        for j in range(len(word)):
            if weight[i][j] > imp:
                if word[j] not in importain_word:
                    importain_word[word[j]] = 1
                else:
                    importain_word[word[j]] += 1
    print(len(importain_word))
    dict_importain_word2= {}
    for var in importain_word:
        if(importain_word[var] > 1):
            dict_importain_word2[var] = importain_word[var]
    print(dict_importain_word2)
    print(len(dict_importain_word2))


def split_data_set():
    #f = open('data/BGL_2k.log',mode='r')
    f = open('data/BGL.log',mode='r')
    f_normal = open('data/normal_new',mode='w')
    f_abnormal = open('data/abnormal_new',mode='w')

   # match_temp = match_log_key(line)
    count = 0
    for line in f:
        count += 1
        if count %10000 == 0:
            print(count)
        isAb = line.split()[0]
        key = match_log_key(line)
        #print(key)
        #print(match_temp[key])
        if isAb =='-':
            f_normal.write(str(key)+'\n')
        else:
            f_abnormal.write(str(key)+'\n')
    return 0

if __name__ == "__main__":
    #count_logkey_in_train()
    predict_all()
    #split_log_by_time()
    #logfile_to_logkeyfile()

    #inputs,outputs,flag = preposess_time_slide('2005-06-03-15.43.1')
    #TP ,FP =  predicted(inputs,outputs,flag)
    #print("TP={} FP={}".format(TP,FP))
    #predict_time_slide('2005-06-03-16.33.3')
    #cout_normal()
    #match_log_key()
    #split_data_set()
#    inputs,outputs = generate('normal')
#    generate('abnormal')
