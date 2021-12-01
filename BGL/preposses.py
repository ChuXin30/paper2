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
# from bgl_train import CNNClassifier
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import cosine
import  math
import  csv
from sklearn.utils import shuffle
import  random
import sys

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

def get_log_time(line):
    time = line.split()[4][:18]
    return str(time)

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
        TP ,FP =  predicted(inputs,outputs,flag) #text_cnn
        #TP ,FP =  predict_deeplog(inputs,outputs,flag) #deeplog
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
    count = 0
    with open('data/' + 'normal_new', 'r') as f:
        line_all = ''
        for line in f.readlines():
            line_all += ' '+ line.replace('\n','')
        line_all = tuple(map(int, line_all.strip().split()))
        for var in line_all:
            count += 1
            if var not in dict_logkey:
                dict_logkey[var] = 1
            else:
                dict_logkey[var] += 1

    with open('data/' + 'abnormal_new', 'r') as f:
        line_all = ''
        for line in f.readlines():
            line_all += ' '+ line.replace('\n','')
        line_all = tuple(map(int, line_all.strip().split()))
        for var in line_all:
            count+=1
            if var not in dict_logkey:
                dict_logkey[var] = 1
            else:
                dict_logkey[var] += 1
    print(len(dict_logkey))
    print("count=%d"%(count))
    dict_logkey = sorted(dict_logkey.items(), key=lambda x: x[1], reverse=False)
    print(dict_logkey)
    for var in dict_logkey:
        print("logkey=%d %f"%(var[0],int(var[1])/count))

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

def split_dataset(total_file,p,train_file,test_file):
    total = 4399503
    f = open(total_file,mode='r')
    f_w = open(train_file,mode='w')
    f_t_n = open(test_file,mode='w')
    count = p*total
    i = 0
    for line in f:
        if(i >= count):
            f_t_n.write(line)
        else:
            f_w.write(line)
        i +=1

def count_logkey(file):
    dict_logkey = {}
    count = 0
    with open(file, 'r') as f:
        line_all = ''
        for line in f.readlines():
            line_all += ' '+ line.replace('\n','')
        line_all = tuple(map(int, line_all.strip().split()))
        for var in line_all:
            count += 1
            if var not in dict_logkey:
                dict_logkey[var] = 1
            else:
                dict_logkey[var] += 1
    print(len(dict_logkey))
    print("count=%d"%(count))
    dict_logkey = sorted(dict_logkey.items(), key=lambda x: x[1], reverse=False)
    print(dict_logkey)
    for var in dict_logkey:
        print("logkey=%d %f"%(var[0],int(var[1])/count))

def word2vec_train(data_file,word_vec,save_model):
    num_features = 100  # Word vector dimensionality
    min_word_count = 1  # Minimum word count
    num_workers = 16  # Number of threads to run in parallel
    context = 4  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    f = open(word_vec, mode='w')


    sentences = word2vec.Text8Corpus(data_file)
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                              size=num_features, min_count=min_word_count, \
                              window=context, sg=1, sample=downsampling)
    model.init_sims(replace=True)
    # 保存模型，供日後使用
    model.save(save_model)

    #    print(model)
    print(model)
    for key in model.wv.vocab.keys():
        x = model[key].tolist()
        list_x = []
        for var in x:
            list_x.append(str(var))
        list_x = " ".join(list_x)
        f.write(key + ' ' + list_x + '\n')

    # 可以在加载模型之后使用另外的句子来进一步训练模型
    # model = gensim.models.Word2Vec.load('/tmp/mymodel')
    # model.train(more_sentences)
    return 0

def split_data(source_file,mode_int ,save_file):
    f = open(source_file,mode='r')
    f_w = open(save_file,mode='w')

    count = 0
    for line in f:
        line = line.replace('\n',' ')
        r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = re.sub(r1, '' , line).lower()
        line2 = ''
        for var in line.split(' '):
            line2 += ' '+var
        if(count %mode_int == 0):
            f_w.write(line2+'\n')
        count+=1

def split_data_top(source_file,top ,save_file):
    f = open(source_file,mode='r')
    f_w = open(save_file,mode='w')
    total = 4399503
    count = 0
    for line in f:
        line = line.replace('\n',' ')
        r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = re.sub(r1, '' , line).lower()
        line2 = ''
        for var in line.split(' '):
            line2 += ' '+var
        if(count < total*top/100):
            f_w.write(line2+'\n')
        count+=1

def get_full_wordvec(col_head,word2vec_100d):
    col = open(col_head,mode='r')
    word2vec_100d = KeyedVectors.load_word2vec_format(word2vec_100d, binary=False)
    dict_word_not_in ={}
    dict_word_in = {}
    for line in col:
        line = line.replace('\n',' ')
        r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^`{|}~]+'
        line = re.sub(r1, ' ' , line).lower()
        line2 = ''
        for var in line.split(' '):
            if var not in word2vec_100d:
                if var not in dict_word_not_in:
                    dict_word_not_in[var] = 1
                else:
                    dict_word_not_in[var] += 1
            else:
                dict_word_in[var] = 1

    print(len(dict_word_in))
    print(len(dict_word_not_in))
    print(dict_word_not_in)
    print(dict_word_in)

def get_importain_word_Tf_idf(word2vec,name,glove_file,top_k,save_file,save_tmp_file):
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec, binary=False)
    glove_model = KeyedVectors.load_word2vec_format(glove_file, binary=False)
    f_s_tmp = open(save_tmp_file,mode='w')
    fr = open(name,mode='r')
    corpus = []

    for line in fr:
        line = line.replace('\n',' ')
        r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = re.sub(r1, ' ' , line).lower()
        line2 = ''
        for var in line.split(' '):
            line2 += ' '+var

        corpus.append(line2)
    #将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer(lowercase=False)
    #计算个词语出现的次数
    X = vectorizer.fit_transform(corpus)
    #获取词袋中所有文本关键词
    word = vectorizer.get_feature_names()
    print(word)


    #查看词频结果
  #  print(X.toarray())
    # ----------------------------------------------------
    #类调用
    transformer = TfidfTransformer()
   # print(transformer)
    #将词频矩阵X统计成TF-IDF值
    tfidf = transformer.fit_transform(X)
    #查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
    print(tfidf.toarray())
    weight = tfidf.toarray()
    print(len(word))
    print(len(weight))
    print(len(weight[0]))
    print(type(weight))
    word_not_in_word2vec_mode = {}
   # top_k = 3
    fw = open(save_file,mode='w')
    log2vec = []
    for i in range(len(weight)):
        top_k_idx = weight[i].argsort()[::-1][0:top_k]
        s = ''
        event_vector = ''
        count = 0
        s_w = ''
        for j in top_k_idx:
            #print(word[j]+' '+str(weight[i][j]))
            s +=' '+ str(word[j])
            s_w += ' ' + str(weight[i][j])

            if word[j] in word2vec_model:
                x = word2vec_model[word[j]].tolist()
            elif word[j] in glove_model:
                x = glove_model[word[j]].tolist()
            else:
                print("word not in word2vec model {}".format(word[j]))
                word_not_in_word2vec_mode[word[j]] = 1
                x = [0]*100

           # x = word[j]
            #print(x)
            #print(type(x))
            #print(type(x[0]))
            #print(weight[i][j])
            #print(type(float(weight[i][j])))
            #for i in range(len(x)):
            #    print(x[i])
            #x = [x[k]* float(weight[i][j]) for k in range(len(x))] # mutiply tf-idf
            list_x = []
            for var in x:
                list_x.append(str(var))
            list_x = " ".join(list_x)
            event_vector += list_x
            if count < top_k-1:
                #print(j)
                event_vector += ' '
            count += 1
        log2vec.append(str(i+1) + ' ' + event_vector + '\n')
        #fw.write(str(i+1) + ' ' + event_vector + '\n')
        print(str(i+1)+' '+s+' '+s_w)
        f_s_tmp.write(str(i+1)+' '+s+' '+s_w+'\n')
    
    fw.write(str(len(log2vec))+' '+str(top_k*100)+'\n')
    for line in log2vec:
        fw.write(line)

    print(word_not_in_word2vec_mode)
    print(len(word_not_in_word2vec_mode))
    '''
    importain_word = {}
    for i in range(len(weight)):
        for j in range(len(word)):
            if weight[i][j] > imp:
                #print(word[j])
                #print(weight[i][j])
                if word[j] not in importain_word:
                    importain_word[word[j]] = 1
                else:
                    importain_word[word[j]] += 1
   # print(importain_word)
   # print(len(importain_word))
    dict_importain_word2= {}

    for var in importain_word:
        if(importain_word[var] > 1):
            dict_importain_word2[var] = importain_word[var]
#            print("%s %f"%{word[j],weight[i][j]})
    #print(dict_importain_word2)
    #print(len(dict_importain_word2))
    '''

def get_importain_word_Tf_idf_with_level(word2vec,name,glove_file,top_k,save_file,save_tmp_file):
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec, binary=False)
    glove_model = KeyedVectors.load_word2vec_format(glove_file, binary=False)
    f_s_tmp = open(save_tmp_file,mode='w')
    fr = open(name,mode='r')
    corpus = []

    for line in fr:
        line = line.replace('\n',' ')
        r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = re.sub(r1, ' ' , line).lower()
        line2 = ''
        for var in line.split(' '):
            line2 += ' '+var

        corpus.append(line2)
    #将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer(lowercase=False)
    #计算个词语出现的次数
    X = vectorizer.fit_transform(corpus)
    #获取词袋中所有文本关键词
    word = vectorizer.get_feature_names()
    print(word)

    #查看词频结果
  #  print(X.toarray())
    # ----------------------------------------------------
    #类调用
    transformer = TfidfTransformer()
   # print(transformer)
    #将词频矩阵X统计成TF-IDF值
    tfidf = transformer.fit_transform(X)
    #查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
    print(tfidf.toarray())
    weight = tfidf.toarray()
    print(len(word))
    print(len(weight))
    print(len(weight[0]))
    print(type(weight))
    word_not_in_word2vec_mode = {}
   # top_k = 3
    fw = open(save_file,mode='w')
    log2vec = []
    for i in range(len(weight)):
        event_vector = ''
        s = ''
        s_w = ''
        #print(weight[i])
        index = weight[i].argsort()[::-1]
        x = [0] * 100
        s += ' ' + ' '
        s_w += ' ' + '1'

        for j in index:
            if 'fatal' == word[j] and weight[i][j] != 0:
                s =' '+ 'fatal'
                s_w = ' ' +'1'
                x = word2vec_model['fatal'].tolist()
            elif 'info' == word[j] and weight[i][j] != 0:
                s =' '+ 'info'
                s_w = ' ' +'1'
                x = word2vec_model['info'].tolist()
        list_x = []
        for var in x:
            list_x.append(str(var))
        list_x = " ".join(list_x)
        event_vector += list_x
        event_vector += ' '
        top_k_idx = weight[i].argsort()[::-1][0:top_k]
        count = 0
        for j in top_k_idx:
            #print(word[j]+' '+str(weight[i][j]))
            s +=' '+ str(word[j])
            s_w += ' ' + str(weight[i][j])

            if word[j] in word2vec_model:
                x = word2vec_model[word[j]].tolist()
            elif word[j] in glove_model:
                x = glove_model[word[j]].tolist()
            else:
                print("word not in word2vec model {}".format(word[j]))
                word_not_in_word2vec_mode[word[j]] = 1
                x = [0]*100

            list_x = []
            for var in x:
                list_x.append(str(var))
            list_x = " ".join(list_x)
            event_vector += list_x
            if count < top_k-1:
                #print(j)
                event_vector += ' '
            count += 1
        log2vec.append(str(i+1) + ' ' + event_vector + '\n')
        #fw.write(str(i+1) + ' ' + event_vector + '\n')
        print(str(i+1)+' '+s+' '+s_w)
        f_s_tmp.write(str(i+1)+' '+s+' '+s_w+'\n')

    fw.write(str(len(log2vec))+' '+str((top_k+1)*100)+'\n')
    for line in log2vec:
        fw.write(line)

    print(word_not_in_word2vec_mode)
    print(len(word_not_in_word2vec_mode))


def save_logkey_in_dataset_normal(file_source,file_save):
    dict_logkey_in_f = {}
    f = open(file_source,mode='r')
    f_save = open(file_save,mode='w')
    for line in f:
        line = line.replace('\n','')
        logkey = int(line)
        dict_logkey_in_f[logkey] = 1
    for var in dict_logkey_in_f:
        f_save.write(str(var)+'\n')

def save_logkey_in_dataset_test(file_source1,file_source2,file_save):
    dict_logkey_in_f = {}
    f1 = open(file_source1,mode='r')
    f2 = open(file_source2,mode='r')

    f_save = open(file_save,mode='w')
    for line in f1:
        line = line.replace('\n','')
        logkey = int(line)
        dict_logkey_in_f[logkey] = 1
    for line in f2:
        line = line.replace('\n','')
        logkey = int(line)
        dict_logkey_in_f[logkey] = 1
    for var in dict_logkey_in_f:
        f_save.write(str(var)+'\n')

def map_testdataset_logkey_to_traindataset_logkey(train_file,test_file,logkey2vec_file,map_file):
    f_train = open(train_file,mode='r')
    f_test = open(test_file,mode='r')
    logkey2vec_model = KeyedVectors.load_word2vec_format(logkey2vec_file, binary=False)
    f_map = open(map_file,mode='w')

    dict_train = {}
    for var in f_train:
        dict_train[var] =1
    print(dict_train)

    for test_k in f_test:
        test_k = test_k.replace('\n','')
        test_k_v = logkey2vec_model[test_k]
        dict_similarity = {}
        for train_k in dict_train:
            train_k = train_k.replace('\n', '')
            train_k_v = logkey2vec_model[train_k]
            s =  cosine(test_k_v, train_k_v)
            #s = math.angle_between(test_k_v, train_k_v)
            dict_similarity[train_k] = s
            #print("%s %s %f"%(test_k,train_k,s))
        #print(dict_similarity)

        print(test_k)
        dict_similarity = sorted(dict_similarity.items(), key=lambda d: d[1])
        print(dict_similarity)
        f_map.write(test_k+' ' +str(dict_similarity[0][0])+'\n')

def unknown_logkey_to_old_logkey(map_file,file_sour,file_save):
    map_f = open(map_file,mode='r')
    source_f = open(file_sour,mode='r')
    save_f = open(file_save,mode='w')

    map_dict = {}
    for map_line in map_f:
        #print(map_line)
        map_list = map_line.replace('\n','').split(' ')
        #print(map_list)
        map_dict[map_list[0]] = map_list[1]
   # print(map_dict)

    for line in source_f:
        var_old = line.replace('\n','')
        var_new = map_dict[var_old]
        save_f.write(var_new+'\n')

def unknown_logkey_to_0(train_logkey,file_sour,file_save):
    f_train = open(train_logkey,mode='r')
    f_s = open(file_sour,mode='r')
    f_save = open(file_save,mode='w')

    dict_train_logkey = {}
    for line in f_train:
        key = line.replace('\n','')
        dict_train_logkey[key] = 1

    for line in f_s:
        line = line.replace('\n','')
        if line in dict_train_logkey:
            f_save.write(line+'\n')
        else:
            f_save.write('0'+'\n')

def dimantion_of_logket2vec(file):
    logkey2vec_model = KeyedVectors.load_word2vec_format(file, binary=False)
    for word, vector in zip(logkey2vec_model.vocab, logkey2vec_model.vectors):
        print(len(vector))

def divide_normal_abnormal(name,noral_csv,abnormal_csv):
    normal_logkey = []
    abnormal_logkey = []
    with open(name) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] == '-':
                normal_logkey.append([row[11]])
            else:
                abnormal_logkey.append([row[11]])

    with open(noral_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入多行数据
        writer.writerows(normal_logkey)

    with open(abnormal_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入多行数据
        writer.writerows(abnormal_logkey)

def create_mapfile_logkey(templates_csv,map_file):
    f_w = open(map_file,'w')
    with open(templates_csv) as f:
        reader = csv.reader(f)
        count = 1
        for row in reader:
            if row[0] != 'EventId':
                f_w.write(row[0]+' '+str(count)+'\n')
                count+=1

def map_logkey_to_num(map_file,source_file,save_file):
    dict_map = {}
    map_f = open(map_file,mode='r')
    source_f = open(source_file,mode='r')
    save_f = open(save_file,mode='w')
    for line in map_f:
        line = line.replace('\n','').split(' ')
      #  print(line)
        dict_map[line[0]] = line[1]
    #print(dict_map)

    for line in source_f:
        line = line.replace('\n','')
        save_f.write(dict_map[line]+'\n')

def csvtemplates_to_templates(map_file,csv_templates,templates):
    dict_map = {}
    map_f = open(map_file,mode='r')
    templates_f = open(templates,mode='w')
    for line in map_f:
        line = line.replace('\n','').split(' ')
        dict_map[line[0]] = line[1]
    with open(csv_templates) as f:
        reader = csv.reader(f)
        for var in reader:
            if var[0] in dict_map:
                logkey = dict_map[var[0]]
                templates_f.write(logkey+','+var[1]+'\n')
            else:
                print(var[0])

def splite_top_dataset(p,normal,abnormal,trian,test_normal,test_abnormal):
    f_normal = open(normal,mode='r')
    f_abnormal = open(abnormal,mode='r')
    f_trian = open(trian,mode='w')
    f_test_normal = open(test_normal,mode='w')
    f_test_abnormal = open(test_abnormal,mode='w')

    list_train = []
    for line in f_normal:
        list_train.append(line)
    length_normal = len(list_train)
    print('total length in normal dataset=%d'%(length_normal))

    for i in range(len(list_train)):
        if i < p*length_normal:
            f_trian.write(list_train[i])
        else:
            f_test_normal.write(list_train[i])

    for line in f_abnormal:
        f_test_abnormal.write(line)

def loganomaly_dataset(window_size,name,save_name):
    f_save = open(save_name,mode='w')
    line_all = []

    with open(name, 'r') as f:
        s = ''
        for line in f.readlines():
            s += ' '+ line.replace('\n','')
        line_all = s.strip().split(' ')
    #print(line_all)
    #print('111111')
    for i in range(len(line_all) - window_size):
        if i % (window_size+1) == 0:
            list_s = line_all[i:i + window_size]
            s = ''
            for var in list_s:
                s+= ' '+str(var)
            s+= ' '+ line_all[i + window_size]
            f_save.write(s+'\n')

def count_logkey_in_trian_test(train,test_normal,test_abnormal,save):
    f_train = open(train,mode='r')
    f_test_normal = open(test_normal,mode='r')
    f_test_abnormal = open(test_abnormal,mode='r')
    f_w = open(save,mode='w')
    train_logkey = {}
    test_abnormal_logkey = {}
    test_normal_logkey = {}
    for line in f_train:
        logkey = line.replace('\n','')
        if logkey in train_logkey:
            train_logkey[logkey] += 1
        else:
            train_logkey[logkey] = 1

    for line in f_test_abnormal:
        logkey = line.replace('\n','')
        if logkey in test_abnormal_logkey:
            test_abnormal_logkey[logkey] += 1
        else :
            test_abnormal_logkey[logkey] = 1

    for line in f_test_normal:
        logkey = line.replace('\n','')
        if logkey in test_normal_logkey:
            test_normal_logkey[logkey] += 1
        else:
            test_normal_logkey[logkey] = 1

    logkey_not_in_train = {}
    count_train_not_in_test_noraml = 0
    for var in test_normal_logkey:
        if var not in train_logkey:
            logkey_not_in_train[var] = test_normal_logkey[var]
            count_train_not_in_test_noraml+= test_normal_logkey[var]

    logkey_not_in_abnormal = {}
    logkey_in_train_and_abnorml={}
    for var in test_abnormal_logkey:
        if var not in train_logkey:
            logkey_not_in_abnormal[var] = test_abnormal_logkey[var]
        elif var in train_logkey:
            logkey_in_train_and_abnorml[var] =test_abnormal_logkey[var]

    f_w.write('train_logkey length'+str(len(train_logkey))+'\n')
    f_w.write(str(train_logkey)+'\n')
    f_w.write('test_normal_logkey length'+str(len(test_normal_logkey))+'\n')
    f_w.write(str(test_normal_logkey)+'\n')
    f_w.write('logkey_not_in_train length'+str(len(logkey_not_in_train))+' '+ str(count_train_not_in_test_noraml)+'\n')
    f_w.write(str(logkey_not_in_train)+'\n')
    f_w.write('test_abnormal_logkey length'+str(len(test_abnormal_logkey))+'\n')
    f_w.write(str(test_abnormal_logkey)+'\n')
    f_w.write('logkey_not_in_abnormal length'+str(len(logkey_not_in_abnormal))+'\n')
    f_w.write(str(logkey_not_in_abnormal)+'\n')
    f_w.write('logkey_in_train_and_abnorml length'+str(len(logkey_in_train_and_abnorml))+'\n')
    f_w.write(str(logkey_in_train_and_abnorml)+'\n')

def random_split_dataset(p, normal, abnormal, trian, test_normal, test_abnormal):
    f_normal = open(normal, mode='r')
    f_abnormal = open(abnormal, mode='r')
    f_trian = open(trian, mode='w')
    f_test_normal = open(test_normal, mode='w')
    f_test_abnormal = open(test_abnormal, mode='w')

    list_train = []
    for line in f_normal:
        list_train.append(line)
    length_normal = len(list_train)
    print('total length in normal dataset=%d' % (length_normal))

    list_train = np.array(list_train)
    indexes = shuffle(np.arange(list_train.shape[0]))
    #print(indexes)
    list_train = list_train[indexes]


    for i in range(len(list_train)):
        if i < p * length_normal:
            f_trian.write(list_train[i])
        else:
            f_test_normal.write(list_train[i])

    for line in f_abnormal:
        f_test_abnormal.write(line)

def random_loganormaly(p,train,test_normal,test_abnormal,save_train,save_test_normal,save_test_abnormal):
    f_train = open(train,mode='r')
    f_test_normal = open(test_normal,mode='r')
    f_test_abnormal = open(test_abnormal,mode='r')
    f_save_train = open(save_train,mode='w')
    f_save_test_normal = open(save_test_normal,mode='w')
    f_save_test_abnormal = open(save_test_abnormal,mode='w')

    normal_dataset = []
    abnormal_dataset = []
    for line in f_train:
        normal_dataset.append(line.replace('\n',''))

    for line in f_test_normal:
        normal_dataset.append(line.replace('\n',''))

    for line in f_test_abnormal:
        abnormal_dataset.append(line.replace('\n',''))

    normal_dataset = np.array(normal_dataset)
    indexes = shuffle(np.arange(normal_dataset.shape[0]))
    normal_dataset = normal_dataset[indexes]

    abnormal_dataset = np.array(abnormal_dataset)
    indexes = shuffle(np.arange(abnormal_dataset.shape[0]))
    abnormal_dataset = abnormal_dataset[indexes]

    length_normal = len(normal_dataset)
    for i in range(len(normal_dataset)):
        if i < p * length_normal:
            f_save_train.write(normal_dataset[i].strip() +'\n')
        else:
            f_save_test_normal.write(normal_dataset[i].strip()+'\n')

    for line in abnormal_dataset:
        f_save_test_abnormal.write(line.strip()+'\n')

def loganormaly_dataset_to_deeplog_dataset(train,test_normal,test_abnormal,save_train,save_test_normal,save_test_abnormal):
    f_train = open(train,mode='r')
    f_test_normal = open(test_normal,mode='r')
    f_test_abnormal = open(test_abnormal,mode='r')
    f_save_train = open(save_train,mode='w')
    f_save_test_normal = open(save_test_normal,mode='w')
    f_save_test_abnormal = open(save_test_abnormal,mode='w')

    train_normal_dataset = []
    test_normal_dataset = []
    abnormal_dataset = []
    for line in f_train:
        line = line.replace('\n','').split(' ')
        for var in line:
            train_normal_dataset.append(var)

    for line in f_test_normal:
        line = line.replace('\n','').split(' ')
        for var in line:
            test_normal_dataset.append(var)

    for line in f_test_abnormal:
        line = line.replace('\n','').split(' ')
        for var in line:
            abnormal_dataset.append(var)

    for line in train_normal_dataset:
        f_save_train.write(line.strip()+'\n')

    for line in test_normal_dataset:
        f_save_test_normal.write(line.strip()+'\n')

    for line in abnormal_dataset:
        f_save_test_abnormal.write(line.strip()+'\n')

def data_set_exist_0(original,save):
    f_or = open(original,mode='r')
    f_s = open(save,mode='w')
    for line in f_or:
        line = line.replace('\n','')
        if line != '0':
            f_s.write(line+'\n')

def group_drain(p,strcuct_csv,map,train,test_noraml,test_abnormal):
    file_trian = open(train,mode='w')
    file_test_normal = open(test_noraml,mode='w')
    file_test_abnormal = open(test_abnormal,mode='w')
    file_map = open(map,mode='r')
    
    normal_logkey = []
    abnormal_logkey = []
    map_dict ={}

    for line in file_map:
        line = line.replace('\n','').split(' ')
        map_dict[line[0]] = line[1]

    #print(map_dict)
    with open(strcuct_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] == '-':
                normal_logkey.append([row[10]])
            else:
                abnormal_logkey.append([row[10]])
    total_length_normal = len(normal_logkey)
    for i in range(total_length_normal):
        if i < p*total_length_normal:
            logkey = str(normal_logkey[i][0])
            if logkey in map_dict:
                logkey = map_dict[logkey]
                file_trian.write(logkey+'\n')
            else:
                print(logkey)
        else:
            logkey = str(normal_logkey[i][0])
            if logkey in map_dict:
                logkey = map_dict[logkey]
                file_test_normal.write(logkey+'\n')
            else:
                print(logkey)

    for var in abnormal_logkey:
        logkey = str(var[0])
        if logkey in map_dict:
            logkey = map_dict[logkey]
            file_test_abnormal.write(logkey+'\n')
        else:
            print(logkey)

def simple_test_dataset(p,window,test_normal,simple_test_normal):
    file_test_normal = open(test_normal,mode='r')
    file_simple_test_normal = open(simple_test_normal,mode='w')

    total_data =[]

    list_total = []
    for line in file_test_normal:
        list_total.append(line.replace('\n',''))

    for i in range(len(list_total)):
        if i%(window+1) == 0:
            total_data.append(list_total[i:i+window+1])

    total_data = np.array(total_data)
    indexes = shuffle(np.arange(total_data.shape[0]))
    total_data = total_data[indexes]

    length_normal = len(total_data)
    print('len simple =%d'%(int(length_normal*p)))
    for i in range(length_normal):
        if i < p * length_normal:
            line = total_data[i]
           # print(line)
            for var in line:
                file_simple_test_normal.write(var+'\n')

def find_normal_abnormal_log_different(strcuct_csv,map,train,test_abnormal,save_file):
    f_train = open(train,mode='r')
    f_test_abnormal = open(test_abnormal,mode='r')
    f_save_file = open(save_file,mode='w')
    train_logkey = {}
    abnormal_logkey = {}
    file_map = open(map,mode='r')

    for line in f_train:
        logkey = line.replace('\n','')
        if logkey in train_logkey:
            train_logkey[logkey] += 1
        else:
            train_logkey[logkey] = 1

    for line in f_test_abnormal:
        logkey = line.replace('\n','')
        if logkey in abnormal_logkey:
            abnormal_logkey[logkey] += 1
        else :
            abnormal_logkey[logkey] = 1

    logkey_not_in_abnormal = {}
    logkey_in_train_and_abnorml={}
    for var in abnormal_logkey:
        if var not in train_logkey:
            logkey_not_in_abnormal[var] = abnormal_logkey[var]
        else:
            logkey_in_train_and_abnorml[var] =abnormal_logkey[var]
    print(train_logkey)
    print(abnormal_logkey)
    print(logkey_in_train_and_abnorml)
    f_save_file.write(str(train_logkey)+'\n')
    f_save_file.write(str(abnormal_logkey)+'\n')
    f_save_file.write(str(logkey_in_train_and_abnorml)+'\n')

    map_dict ={}
    for line in file_map:
        line = line.replace('\n','').split(' ')
        map_dict[line[1]] = line[0]
    for logkey in logkey_in_train_and_abnorml:
        logkey_num = logkey
        logkey = map_dict[logkey]
        print(logkey+' '+ str(logkey_in_train_and_abnorml[logkey_num]))
        f_save_file.write(logkey+' '+ logkey_num+' '+str(logkey_in_train_and_abnorml[logkey_num])+'\n')

        with open(strcuct_csv) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[1] == '-' and logkey == row[10]:
                    print(str(row))
                    f_save_file.write(str(row) + '\n')
                    break
        count = 0
        with open(strcuct_csv) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[1] != '-' and logkey == row[10]:
                    f_save_file.write(str(row) + '\n')
                    count+=1
                    if count %5 == 0:
                        break
        f_save_file.write('\n')

def find_logkey_in_train_not_in_test_normal(train,test,save_file):
    f_train = open(train,mode='r')
    f_test_normal = open(test,mode='r')
    f_save_file = open(save_file,mode='w')
    train_logkey = {}
    normal_logkey = {}

    for line in f_train:
        logkey = line.replace('\n','')
        if logkey in train_logkey:
            train_logkey[logkey] += 1
        else:
            train_logkey[logkey] = 1

    for line in f_test_normal:
        logkey = line.replace('\n','')
        if logkey in normal_logkey:
            normal_logkey[logkey] += 1
        else :
            normal_logkey[logkey] = 1

    for var in normal_logkey:
        if var not in train_logkey:
            f_save_file.write('logkey:'+var+' '+'num:'+ str(normal_logkey[var]) +'\n')



def get_word_table(file,save_file):
    f_r = open(file,mode='r')
    f_s = open(save_file,mode='w')
    dict_word = {}

    for line in f_r:
        line = line.replace('\n','').replace('\\',' ')
        r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = re.sub(r1, ' ', line)
        line = line.split(' ')
        for var in line:
            if var not in dict_word:
                dict_word[var] = 1
            else:
                dict_word[var] += 1
    #print(dict_word)
    for var in dict_word:
        if var != '':
            f_s.write(var+'\n')

f_word = open('noise/word_table',mode='r')
list_word = []
for line in f_word:
    line = line.replace('\n','')
    list_word.append(line)

def add_a_word_in_log(s):
    length_w = len(list_word)
    s = s +' '+ list_word[random.randint(0,length_w-1)]

    int_a =  random.randint(0, 100)
    if int_a % 2 == 0:
        s = s +' '+ 'to'
    return  s

f_word_1 = open('noise/word_table',mode='r')
dict_word = {}
for line in f_word_1:
    line = line.replace('\n','')
    dict_word[line] = 1

def del_a_word_in_log(s):
    list_s = s.replace('\n','').split(' ')

    var = len(list_s)-1
    while var >= 0:
        if list_s[var] not in dict_word :
            list_s.remove(list_s[var])
            break
        var= var -1

    s = ' '.join(list_s)
    return  s

def add_noise(sorce_file,p,save_file):
    source_f = open(sorce_file,mode='r')
    save_f = open(save_file,mode='w')
    p_add = p*3.0/3
    p_del = p*1.0/40000

    count = 0
    for line in source_f:
        #print(line)
        t =  random.random()
        line = line.replace('\n','')
        if(t < p_add):
            line = add_a_word_in_log(line)

        t = random.random()
        if(t< p_del):
            line = del_a_word_in_log(line)
        save_f.write(line+'\n')
        #print(line)
        count+=1
        if(count%10000==0):
            print(count)

def get_importain_word_csv(name, top_k,save_tmp_file):
    f_s_tmp = open(save_tmp_file, mode='w')
    fr = open(name, mode='r')
    list_temp = []
    corpus = []
    for line in fr:
        #print(line)
        line1 = line
        #print(line1)
        if line1.replace('\n','')== 'EventId,EventTemplate,Occurrences':
            continue
        line = line.replace('\n', ' ').replace(',', ' ')
        line = line[8:]  # sub head
        #line = replace_file(line)
        r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        #r1 = '[0-9’!"#$%&\'\\()+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

        line = re.sub(r1, ' ', line).lower()
        line2 = ''
        #print(line)

        list_temp.append(line.split(' '))
        for var in line.split(' '):
            line2 += ' ' + var
        corpus.append(line2)
    vectorizer = CountVectorizer(lowercase=False)
    X = vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    weight = tfidf.toarray()

    for i in range(len(weight)):
        top_k_idx = weight[i].argsort()[::-1][0:top_k]
        s = ''
        s_w = ''
        for j in top_k_idx:
            if word[j] in list_temp[i]:
                s += ' ' + str(word[j])
                s_w += ' ' + str(weight[i][j])
        f_s_tmp.write(str(i + 1) + ' ' + s + ' ' + s_w + '\n')

def similartiy_template_0(unknow,exist):
    logkey_unknow = unknow.split(' ')[0]
    logkey_exist = exist.split(' ')[0]
    r1 = '[0-9’!"#$%&\'\\()+,-./:;=?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    unknow = re.sub(r1, '', unknow).lower().strip()
    exist = re.sub(r1, '', exist).lower().strip()

    list_unknow = unknow.split(' ')
    list_exist = exist.split(' ')

    count = 0
    length = 0
    for var in list_unknow:
        if var != '' and var != ' ' and var !='to' and var != 'for' :
            length += 1

    for var in list_exist:
        if var != '' and var != ' ' and var !='to' and var != 'for':
            length += 1

    for unknow_var in list_unknow:
        for exist_var in list_exist:
            if exist_var == unknow_var and exist_var != '' and exist_var !=' ' and exist_var !='to' and exist_var != 'for' and exist_var != '*':
                count += 1
    #print(unknow)
    #print(exist)
    #print(count*2.0/length)
    return logkey_unknow,logkey_exist,count*2.0/length

def similartiy_template(unknow,exist):
    logkey_unknow = unknow.split(' ')[0]
    logkey_exist = exist.split(' ')[0]
    r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    unknow = re.sub(r1, '', unknow).lower().strip()
    exist = re.sub(r1, '', exist).lower().strip()
    list_unknow = unknow.split(' ')
    list_exist = exist.split(' ')
    count = 0
    length = 0
    for var in list_unknow:
        if var != '' and var != ' ' and var !='to' and var != 'for' :
            length += 1

    for var in list_exist:
        if var != '' and var != ' ' and var !='to' and var != 'for':
            length += 1

    for unknow_var in list_unknow:
        for exist_var in list_exist:
            if exist_var == unknow_var and exist_var != '' and exist_var !=' ' and exist_var !='to' and exist_var != 'for':
                count += 1
            if exist_var == unknow_var and exist_var == 'info' or exist_var == unknow_var and exist_var == 'fatal' or exist_var == unknow_var and exist_var == 'warning' or exist_var == unknow_var and exist_var == 'failure' :
                count += length
    return logkey_unknow,logkey_exist,count*2.0/length

def get_map_unkonw_tmp_to_exist(exist_t,unknow_t,save_map,save_temp_si):
    f_exist = open(exist_t,mode='r')
    f_unknow = open(unknow_t,mode='r')
    f_map = open(save_map,mode='w')
    f_save = open(save_temp_si,mode='w')

    list_exist = []

    for line in f_exist:
        list_exist.append(line)

    f_map.write('')
    for line in f_unknow:
        dict_similarity = {}
        for exist in list_exist:
            logkey_unknow,logkey_exist, value = similartiy_template(line,exist)
            dict_similarity[logkey_exist] = value
        #print(dict_similarity)
        dict_similarity = sorted(dict_similarity.items(), key=lambda d: d[1] ,reverse=True )
        f_save.write(line)
        f_save.write(str(dict_similarity)+'\n')
       # print(dict_similarity[0][1])
        f_map.write(line.split(' ')[0]+' ' +str(dict_similarity[0][0])+'\n')

def get_map_unkonw_tmp_to_0(exist_t,th,unknow_t,save_map):
    f_exist = open(exist_t,mode='r')
    f_unknow = open(unknow_t,mode='r')
    f_map = open(save_map,mode='w')
    list_exist = []

    for line in f_exist:
        list_exist.append(line)

    f_map.write('')
    for line in f_unknow:
        dict_similarity = {}
        for exist in list_exist:
            logkey_unknow,logkey_exist, value = similartiy_template_0(line,exist)
            dict_similarity[logkey_exist] = value
        #print(dict_similarity)
        dict_similarity = sorted(dict_similarity.items(), key=lambda d: d[1] ,reverse=True )
        #print(dict_similarity[0][1])
        if dict_similarity[0][1] >= th:
            f_map.write(line.split(' ')[0]+' ' +str(dict_similarity[0][0])+'\n')
        else:
            f_map.write(line.split(' ')[0] + ' 0\n')

def get_map_0_to_exist(p_num, map_0, map_exist, map_0_new):
    f_map_0 = open(map_0, mode='r')
    f_map_exist = open(map_exist, mode='r')
    f_map_0_new = open(map_0_new, mode='w')
    dict_map_0 = {}
    dict_map_exist = {}
    total_0 = 0
    total = 0
    list_map_0 = []
    for line in f_map_0:
        line = line.replace('\n', '').split(' ')
        dict_map_0[line[0]] = line[1]
        list_map_0.append(line)
        if line[1] == '0':
            total_0 += 1
        total += 1

    for line in f_map_exist:
        line = line.replace('\n', '').split(' ')
        dict_map_exist[line[0]] = line[1]

    i = 0
    dict_map_0_new = {}
    for var in dict_map_0:
        if dict_map_0[var] == '0' and i < p_num * total_0:
            dict_map_0_new[var] = dict_map_exist[var]
            i += 1
        else:
            dict_map_0_new[var] = dict_map_0[var]
    for var in dict_map_0_new:
        f_map_0_new.write(var + ' ' + dict_map_0_new[var] + '\n')

def count_deeplog_0(normal_f,abnormal_f):
    f_n = open(normal_f,mode='r')
    f_ab = open(abnormal_f,mode='r')

    count_logkey = {}
    total = 0
    total_0 = 0
    for line in f_n:
        line = line.replace('\n','').split(' ')
        total += 1
        #print(line)
        for var in line:
            #print(var)
            if var == '0':
                total_0 += 1
                break
    for line in f_ab:
        line = line.replace('\n','').split(' ')
        total += 1
        for var in line:
            if var == '0':
                total_0 += 1
                break

    print("total_line=%d  total_line_with_0=%d %f"%(total,total_0,total_0*100.0/total))

def find_logkey_normal_abnormal(bgl_csv,map_logkey,map_file,template_csv):
    logkey_normal = {}
    logkey_abnormal = {}
    f_save = open(map_file,mode='w')
    f_map = open(map_logkey,mode='r')

    dict_regex = {}
    with open(template_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] != 'EventId':
                dict_regex[row[0]] = row[1]

    dict_map = {}
    for line in f_map:
        line = line.replace('\n','').split(' ')
        dict_map[line[0]] = line[1]

    with open(bgl_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            logkey = row[10]
            if logkey == 'EventId':
                continue
            if logkey not in logkey_normal:
                if row[1] == '-':
                    logkey_normal[logkey] = 1
                    logkey_abnormal[logkey] = 0
                else:
                    logkey_normal[logkey] = 0
                    logkey_abnormal[logkey] = 1
            else:
                if row[1] == '-':
                    logkey_normal[logkey] += 1
                else:
                    logkey_abnormal[logkey] += 1

    for var in logkey_normal:
        if var not in dict_map:
            continue
        if logkey_normal[var] > logkey_abnormal[var]:
            f_save.write(var+' '+dict_map[var] +' '+'0'+' '+str(logkey_normal[var])+' '+str(logkey_abnormal[var])+' '+dict_regex[var] +'\n')
        else:
            f_save.write(var+' '+dict_map[var] +' '+'1'+' '+str(logkey_normal[var])+' '+str(logkey_abnormal[var])+' '+dict_regex[var] +'\n')

def find_error_match(new,old,map,save):
    f_new  = open(new,mode='r')
    f_old = open(old,mode='r')
    f_map = open(map,mode='r')
    f_save = open(save,mode='w')

    dict_map = {}
    for line in f_map:
        line = line.replace('\n','').split(' ')
        dict_map[line[0]] = line[1]

    new_logkey_lable = {}
    new_logkey_num = {}
    for line in f_new:
        line = line.replace('\n','').split(' ')
        #print(line)
        new_logkey_lable[line[1]] = line[2]

        if line[2] == '0':
            new_logkey_num[line[1]] = line[3]
        else:
            new_logkey_num[line[1]] = line[4]
        if line[1] == '544':
            print(line)
            print(new_logkey_num[line[1]])

    old_logkey_lable = {}
    old_logkey_num = {}
    for line in f_old:
        line = line.replace('\n','').split(' ')
        old_logkey_lable[line[1]] = line[2]
        if line[2] == '0':
            old_logkey_num[line[1]] = line[3]
        else:
            old_logkey_num[line[1]]  = line[4]

    for var in dict_map:
        old_logeky = dict_map[var]
        if new_logkey_lable[var] != old_logkey_lable[old_logeky]:
            f_save.write('logkey:'+var+' lable:'+new_logkey_lable[var]+' num:'+ new_logkey_num[var]+' old_logkey:' + old_logeky+' lable:'+ old_logkey_lable[old_logeky]+' num:'+ old_logkey_num[old_logeky]+'\n')



if __name__ == "__main__":
    dir = sys.argv[1]
    loganomaly_dataset(20, dir + '/WITH_S_train', dir + '/hdfs_train_normal')
    loganomaly_dataset(20, dir + '/Deeplog_test_normal_simple', dir + '/hdfs_test_normal_simple')
    loganomaly_dataset(20, dir + '/Deeplog_test_abnormal_simple', dir + '/hdfs_test_abnormal_simple')
    loganomaly_dataset(20, dir + '/Deeplog_test_normal', dir + '/hdfs_test_normal')
    loganomaly_dataset(20, dir + '/Deeplog_test_abnormal', dir + '/hdfs_test_abnormal')
    # get_importain_word_Tf_idf_with_level('data/word2vec_model_100d', 'data/drain_random_with_level_20/templates', 'data/glove.6B.100d.word2vec.txt',4,
    #                           dir+'/event_vector_top4_with_level.txt',dir+'/save_top4_with_levels.txt')
    # dimantion_of_logket2vec('noise/event_vector_top4_with_level.txt')


    # dir = 'noise_2'
    # # 1.1add noise in bgl dataset
    # add_noise('data/BGL.log', 0.30, dir+'/BGL.log')
    # noise_3
    # dir = 'noise_3'
    # # 1.1add noise in bgl dataset
    # add_noise('data/BGL.log', 0.20, dir+'/BGL.log')
    # dir = 'noise_4'
    # # 1.1add noise in bgl dataset
    # add_noise('data/BGL.log', 0.10, dir+'/BGL.log')
    # dir = 'noise_5'
    # # 1.1add noise in bgl dataset
    # add_noise('data/BGL.log', 0.05, dir+'/BGL.log')
    # dir = 'noise_6'
    # add_noise('data/BGL.log', 0.01, dir+'/BGL.log')
    # dir = 'noise_7'
    # add_noise('data/BGL.log', 0.001, dir+'/BGL.log')
    # dir = 'noise_8'
    # add_noise('data/BGL.log', 0.000, dir+'/BGL.log')
    # dir = 'noise_9'
    #add_noise('data/BGL.log', 1, dir+'/BGL.log')
    # dir = 'noise_10'
    # add_noise('data/BGL.log', 0.5, dir+'/BGL.log')
    # 1.2use drain pares noise/bgl.log
    # create_mapfile_logkey(dir+'/BGL.log_templates.csv',dir+'/logkey_num_map') #creat map file such as '070de4aa' map to '1'
    # group_drain(0.8, dir+'/BGL.log_structured.csv',dir+'/logkey_num_map' ,dir+'/temp/train_normal', dir+'/temp/test_normal', dir+'/temp/test_abnormal')
    # loganomaly_dataset(20, dir+'/temp/train_normal', dir+'/temp/hdfs_train_normal')
    # loganomaly_dataset(20, dir+'/temp/test_normal', dir+'/temp/hdfs_test_normal')
    # loganomaly_dataset(20, dir+'/temp/test_abnormal', dir+'/temp/hdfs_test_abnormal')
    # random_loganormaly(0.2, dir+'/temp/hdfs_train_normal', dir+'/temp/hdfs_test_normal', dir+'/temp/hdfs_test_abnormal', dir+'/temp/train_1', dir+'/temp/test_normal_1', dir+'/temp/test_abnormal_1')
    # loganormaly_dataset_to_deeplog_dataset(dir+'/temp/train_1',\
    #                                     dir+'/temp/test_normal_1',\
    #                                     dir+'/temp/test_abnormal_1',\
    #                                     dir+'/train',\
    #                                     dir+'/test_normal',\
    #                                     dir+'/test_abnormal')
    # find_normal_abnormal_log_different(dir+'/BGL.log_structured.csv',dir+'/logkey_num_map',dir+'/train',dir+'/test_abnormal',dir+'/normal_abnormal_differernt')
    # find_logkey_in_train_not_in_test_normal(dir+'/train',dir+'/test_normal',dir+'/not_in_test_normal')
    # dir_noise = dir+'/noise_0'#
    # sim_throad_0 = 1
    # p_num = 0
    # dir_noise = dir+'/noise_5'#
    # sim_throad_0 = 1
    # p_num = 0.083
    # get_importain_word_csv('data/drain_random_with_level_20/BGL.log_templates.csv', 100, dir+'/templates_important_word_old')
    # get_importain_word_csv(dir+'/BGL.log_templates.csv', 100, dir+'/templates_important_word')
    # get_map_unkonw_tmp_to_exist(dir+'/templates_important_word_old', dir+'/templates_important_word',dir_noise+'/map_know_to_exist',dir_noise+'/save_sim')
    # get_map_unkonw_tmp_to_0(dir+'/templates_important_word_old',sim_throad_0,dir+'/templates_important_word',dir_noise+'/map_know_to_0')
    # get_map_0_to_exist(p_num,dir_noise+'/map_know_to_0',dir_noise+'/map_know_to_exist',dir_noise+'/map_know_to_0_new')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', dir+'/train',dir_noise + '/WITH_S_train')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', dir+'/test_normal',dir_noise + '/WITH_S_test_normal')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', dir+'/test_abnormal',dir_noise + '/WITH_S_test_abnormal')
    # find_normal_abnormal_log_different(dir+'/BGL.log_structured.csv',dir+'/logkey_num_map',dir+'/noise_0/WITH_S_train',dir+'/noise_0/WITH_S_test_abnormal',dir+'/noise_0/normal_abnormal_differernt')
    # find_logkey_in_train_not_in_test_normal(dir+'/noise_0/WITH_S_train',dir+'/noise_0/WITH_S_test_abnormal',dir+'/noise_0/not_in_test_normal')
    # find_logkey_normal_abnormal(dir+'/BGL.log_structured.csv',dir+'/logkey_num_map',dir+'/lable_logkey',dir+'/BGL.log_templates.csv')
    # find_logkey_normal_abnormal('struct_data/drain/BGL.log_structured.csv','struct_data/drain/logkey_num_map',dir+'/lable_logkey_row','struct_data/drain/BGL.log_templates.csv')
    # find_error_match(dir+'/lable_logkey',dir+'/lable_logkey_row',dir+'/noise_0/map_know_to_exist',dir+'/noise_0/error')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_0_new', dir + '/test_normal',dir_noise + '/Deeplog_test_normal')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_0_new', dir + '/test_abnormal', dir_noise + '/Deeplog_test_abnormal')
    # simple_test_dataset(0.05, 20, dir_noise + '/WITH_S_test_normal',   dir_noise + '/WITH_S_test_normal_simple')
    # simple_test_dataset(0.05, 20, dir_noise + '/WITH_S_test_abnormal', dir_noise + '/WITH_S_test_abnormal_simple')
    # simple_test_dataset(0.05, 20, dir_noise + '/Deeplog_test_normal',  dir_noise + '/Deeplog_test_normal_simple')
    # simple_test_dataset(0.05, 20, dir_noise + '/Deeplog_test_abnormal',dir_noise + '/Deeplog_test_abnormal_simple')
    # count_deeplog_0(dir_noise + '/Deeplog_test_normal',dir_noise + '/Deeplog_test_abnormal')

 #noise_2
 # 1.1add noise in bgl dataset
 #add_noise('data/BGL.log', 0.3, 'noise_2/BGL.log')
 #1.2use drain pares noise/bgl.log

    # create_mapfile_logkey('noise_2/BGL.log_templates.csv','noise_2/logkey_num_map') #creat map file such as '070de4aa' map to '1'
    # group_drain(0.8, 'noise_2/BGL.log_structured.csv','noise_2/logkey_num_map' ,'noise_2/temp/train_normal', 'noise_2/temp/test_normal', 'noise_2/temp/test_abnormal')
    # loganomaly_dataset(20, 'noise_2/temp/train_normal', 'noise_2/temp/hdfs_train_normal')
    # loganomaly_dataset(20, 'noise_2/temp/test_normal', 'noise_2/temp/hdfs_test_normal')
    # loganomaly_dataset(20, 'noise_2/temp/test_abnormal', 'noise_2/temp/hdfs_test_abnormal')
    # random_loganormaly(0.2, 'noise_2/temp/hdfs_train_normal', 'noise_2/temp/hdfs_test_normal', 'noise_2/temp/hdfs_test_abnormal', 'noise_2/temp/train_1', 'noise_2/temp/test_normal_1', 'noise_2/temp/test_abnormal_1')
    # loganormaly_dataset_to_deeplog_dataset('noise_2/temp/train_1',\
    #                                     'noise_2/temp/test_normal_1',\
    #                                     'noise_2/temp/test_abnormal_1',\
    #                                     'noise_2/train',\
    #                                     'noise_2/test_normal',\
    #                                     'noise_2/test_abnormal')
    # find_normal_abnormal_log_different('noise_2/BGL.log_structured.csv','noise_2/logkey_num_map','noise_2/train','noise_2/test_abnormal','noise_2/normal_abnormal_differernt')
    # find_logkey_in_train_not_in_test_normal('noise_2/train','noise_2/test_normal','noise_2/not_in_test_normal')
    # dir_noise = 'noise_2/noise_0'#
    # sim_throad_0 = 0.99
    # p_num = 1
    # get_importain_word_csv('data/drain_random_with_level_20/BGL.log_templates.csv', 100, 'noise_2/templates_important_word_old')
    # get_importain_word_csv('noise_2/BGL.log_templates.csv', 100, 'noise_2/templates_important_word')
    # get_map_unkonw_tmp_to_exist('noise_2/templates_important_word_old', 'noise_2/templates_important_word',dir_noise+'/map_know_to_exist',dir_noise+'/save_sim')
    # get_map_unkonw_tmp_to_0('noise_2/templates_important_word_old',sim_throad_0,'noise_2/templates_important_word',dir_noise+'/map_know_to_0')
    # get_map_0_to_exist(p_num,dir_noise+'/map_know_to_0',dir_noise+'/map_know_to_exist',dir_noise+'/map_know_to_0_new')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', 'noise_2/train',dir_noise + '/WITH_S_train')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', 'noise_2/test_normal',dir_noise + '/WITH_S_test_normal')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', 'noise_2/test_abnormal',dir_noise + '/WITH_S_test_abnormal')
    # find_normal_abnormal_log_different('noise_2/BGL.log_structured.csv','noise_2/logkey_num_map','noise_2/noise_0/WITH_S_train','noise_2/noise_0/WITH_S_test_abnormal','noise_2/noise_0/normal_abnormal_differernt')
    # find_logkey_in_train_not_in_test_normal('noise_2/noise_0/WITH_S_train','noise_2/noise_0/WITH_S_test_abnormal','noise_2/noise_0/not_in_test_normal')
    # find_logkey_normal_abnormal('noise_2/BGL.log_structured.csv','noise_2/logkey_num_map','noise_2/lable_logkey','noise_2/BGL.log_templates.csv')
    # find_logkey_normal_abnormal('struct_data/drain/BGL.log_structured.csv','struct_data/drain/logkey_num_map','noise_2/lable_logkey_row','struct_data/drain/BGL.log_templates.csv')
    # find_error_match('noise_2/lable_logkey','noise_2/lable_logkey_row','noise_2/noise_0/map_know_to_exist','noise_2/noise_0/error')
    # dir_noise = 'noise_2/noise_0'#
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_0_new', 'noise/test_normal',dir_noise + '/Deeplog_test_normal')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_0_new', 'noise/test_abnormal', dir_noise + '/Deeplog_test_abnormal')
    # simple_test_dataset(0.05, 20, dir_noise + '/WITH_S_test_normal',   dir_noise + '/WITH_S_test_normal_simple')
    # simple_test_dataset(0.05, 20, dir_noise + '/WITH_S_test_abnormal', dir_noise + '/WITH_S_test_abnormal_simple')
    # simple_test_dataset(0.05, 20, dir_noise + '/Deeplog_test_normal',  dir_noise + '/Deeplog_test_normal_simple')
    # simple_test_dataset(0.05, 20, dir_noise + '/Deeplog_test_abnormal',dir_noise + '/Deeplog_test_abnormal_simple')

    #noise_2/noise_15
    # dir_noise = 'noise_2/noise_25'#
    # sim_throad_0 = 1
    # p_num = 0.1
    # get_importain_word_csv('data/drain_random_with_level_20/BGL.log_templates.csv', 100, 'noise_2/templates_important_word_old')
    # get_importain_word_csv('noise_2/BGL.log_templates.csv', 100, 'noise_2/templates_important_word')
    # get_map_unkonw_tmp_to_exist('noise_2/templates_important_word_old', 'noise_2/templates_important_word',dir_noise+'/map_know_to_exist',dir_noise+'/save_sim')
    # get_map_unkonw_tmp_to_0('noise_2/templates_important_word_old',sim_throad_0,'noise_2/templates_important_word',dir_noise+'/map_know_to_0')
    # get_map_0_to_exist(p_num,dir_noise+'/map_know_to_0',dir_noise+'/map_know_to_exist',dir_noise+'/map_know_to_0_new')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', 'noise_2/train',dir_noise + '/WITH_S_train')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', 'noise_2/test_normal',dir_noise + '/WITH_S_test_normal')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', 'noise_2/test_abnormal',dir_noise + '/WITH_S_test_abnormal')
    # find_normal_abnormal_log_different('noise_2/BGL.log_structured.csv','noise_2/logkey_num_map','noise_2/noise_0/WITH_S_train','noise_2/noise_0/WITH_S_test_abnormal','noise_2/noise_0/normal_abnormal_differernt')
    # find_logkey_in_train_not_in_test_normal('noise_2/noise_0/WITH_S_train','noise_2/noise_0/WITH_S_test_abnormal','noise_2/noise_0/not_in_test_normal')
    # find_logkey_normal_abnormal('noise_2/BGL.log_structured.csv','noise_2/logkey_num_map','noise_2/lable_logkey','noise_2/BGL.log_templates.csv')
    # find_logkey_normal_abnormal('struct_data/drain/BGL.log_structured.csv','struct_data/drain/logkey_num_map','noise_2/lable_logkey_row','struct_data/drain/BGL.log_templates.csv')
    # find_error_match('noise_2/lable_logkey','noise_2/lable_logkey_row','noise_2/noise_0/map_know_to_exist','noise_2/noise_0/error')
    # dir_noise = 'noise_2/noise_0'#
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_0_new', 'noise/test_normal',dir_noise + '/Deeplog_test_normal')
    # unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_0_new', 'noise/test_abnormal', dir_noise + '/Deeplog_test_abnormal')
    # simple_test_dataset(0.05, 20, dir_noise + '/WITH_S_test_normal',   dir_noise + '/WITH_S_test_normal_simple')
    # simple_test_dataset(0.05, 20, dir_noise + '/WITH_S_test_abnormal', dir_noise + '/WITH_S_test_abnormal_simple')
    # simple_test_dataset(0.05, 20, dir_noise + '/Deeplog_test_normal',  dir_noise + '/Deeplog_test_normal_simple')
    # simple_test_dataset(0.05, 20, dir_noise + '/Deeplog_test_abnormal',dir_noise + '/Deeplog_test_abnormal_simple')
    # count_deeplog_0(dir_noise + '/Deeplog_test_normal',dir_noise + '/Deeplog_test_abnormal')

    #noise random noise_0
    #1.1add noise in bgl dataset
    #get_word_table('data/BGL.log','noise/word_table')
    #add_noise('data/BGL.log',0.3,'noise/BGL.log')
    #1.2use drain pares noise/bgl.log
    #1.3log parse to logkey and random split dataset
    #create_mapfile_logkey('noise/BGL.log_templates.csv','noise/logkey_num_map') #creat map file such as '070de4aa' map to '1'
    #group_drain(0.8, 'noise/BGL.log_structured.csv','noise/logkey_num_map' ,'noise/temp/train_normal', 'noise/temp/test_normal', 'noise/temp/test_abnormal')
    #loganomaly_dataset(20, 'noise/temp/train_normal', 'noise/temp/hdfs_train_normal')
    #loganomaly_dataset(20, 'noise/temp/test_normal', 'noise/temp/hdfs_test_normal')
    #loganomaly_dataset(20, 'noise/temp/test_abnormal', 'noise/temp/hdfs_test_abnormal')
    #random_loganormaly(0.2, 'noise/temp/hdfs_train_normal', 'noise/temp/hdfs_test_normal', 'noise/temp/hdfs_test_abnormal', 'noise/temp/train_1', 'noise/temp/test_normal_1', 'noise/temp/test_abnormal_1')
    #loganormaly_dataset_to_deeplog_dataset('noise/temp/train_1',\
    #                                    'noise/temp/test_normal_1',\
    #                                    'noise/temp/test_abnormal_1',\
    #                                    'noise/train',\
    #                                    'noise/test_normal',\
    #                                    'noise/test_abnormal')
    #1.4 unknow map to 0 and unknow map to exist
    #dir_noise = 'noise/noise_15'#total_line=3866457  total_line_with_0=564510 14.600188
    #sim_throad_0 = 1
    #p_num = 0.0
    #dir_noise = 'noise/noise_5'#total_line=3866457  total_line_with_0=198427 5.132011
    #sim_throad_0 = 1
    #p_num = 0.1
    #dir_noise = 'noise/noise_10'#total_line=3866457  total_line_with_0=413550 10.695839
    #sim_throad_0 = 0.9
    #p_num = 0.00
    #dir_noise = 'noise/noise_0'#
    #sim_throad_0 = 0.9
    #p_num = 1
    #get_importain_word_csv('data/drain_random_with_level_20/BGL.log_templates.csv', 100, 'data/templates_important_word')
    #get_importain_word_csv('noise/BGL.log_templates.csv', 100, 'noise/templates_important_word')
    #get_map_unkonw_tmp_to_exist('data/templates_important_word', 'noise/templates_important_word',dir_noise+'/map_know_to_exist')
    #get_map_unkonw_tmp_to_0('data/templates_important_word',sim_throad_0,'noise/templates_important_word',dir_noise+'/map_know_to_0')
    #get_map_0_to_exist(p_num,dir_noise+'/map_know_to_0',dir_noise+'/map_know_to_exist',dir_noise+'/map_know_to_0_new')
    #unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', 'noise/train',dir_noise + '/WITH_S_train')
    #unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', 'noise/test_normal',dir_noise + '/WITH_S_test_normal')
    #unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', 'noise/test_abnormal',dir_noise + '/WITH_S_test_abnormal')
    # deeplog
    #unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_0_new', 'noise/test_normal',dir_noise + '/Deeplog_test_normal')
    #unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_0_new', 'noise/test_abnormal', dir_noise + '/Deeplog_test_abnormal')
    #count 0 in test_dataset
    #count_deeplog_0(dir_noise + '/Deeplog_test_normal',dir_noise + '/Deeplog_test_abnormal')
    #simple_test_dataset(0.05, 20, dir_noise + '/WITH_S_test_normal',   dir_noise + '/WITH_S_test_normal_simple')
    #simple_test_dataset(0.05, 20, dir_noise + '/WITH_S_test_abnormal', dir_noise + '/WITH_S_test_abnormal_simple')
    #simple_test_dataset(0.05, 20, dir_noise + '/Deeplog_test_normal',  dir_noise + '/Deeplog_test_normal_simple')
    #simple_test_dataset(0.05, 20, dir_noise + '/Deeplog_test_abnormal',dir_noise + '/Deeplog_test_abnormal_simple')
    #find_logkey_in_train_not_in_test_normal('noise/noise_0/WITH_S_train','noise/noise_0/WITH_S_test_normal','noise/noise/not_in_test_normal')
    #find_normal_abnormal_log_different('noise/BGL.log_structured.csv','noise/logkey_num_map','noise/noise_0/WITH_S_train','noise/noise_0/WITH_S_test_abnormal','noise/noise_0/normal_abnormal_differernt')
    #find_logkey_in_train_not_in_test_normal('noise/train','noise/test_normal','noise/not_in_test_normal')
    #find_normal_abnormal_log_different('noise/BGL.log_structured.csv','noise/logkey_num_map','noise/train','noise/test_abnormal','noise/normal_abnormal_differernt')

    #noise_1
    #1.2use drain pares noise/bgl.log
    #1.3log parse to logkey and random split dataset
    '''
    create_mapfile_logkey('noise_1/BGL.log_templates.csv','noise_1/logkey_num_map') #creat map file such as '070de4aa' map to '1'
    group_drain(0.8, 'noise_1/BGL.log_structured.csv','noise_1/logkey_num_map' ,'noise_1/temp/train_normal', 'noise_1/temp/test_normal', 'noise_1/temp/test_abnormal')
    loganomaly_dataset(20, 'noise_1/temp/train_normal', 'noise_1/temp/hdfs_train_normal')
    loganomaly_dataset(20, 'noise_1/temp/test_normal', 'noise_1/temp/hdfs_test_normal')
    loganomaly_dataset(20, 'noise_1/temp/test_abnormal', 'noise_1/temp/hdfs_test_abnormal')
    random_loganormaly(0.2, 'noise_1/temp/hdfs_train_normal', 'noise_1/temp/hdfs_test_normal', 'noise_1/temp/hdfs_test_abnormal', 'noise_1/temp/train_1', 'noise_1/temp/test_normal_1', 'noise_1/temp/test_abnormal_1')
    loganormaly_dataset_to_deeplog_dataset('noise_1/temp/train_1',\
                                        'noise_1/temp/test_normal_1',\
                                        'noise_1/temp/test_abnormal_1',\
                                        'noise_1/train',\
                                        'noise_1/test_normal',\
                                        'noise_1/test_abnormal')
    find_normal_abnormal_log_different('noise_1/BGL.log_structured.csv','noise_1/logkey_num_map','noise_1/train','noise_1/test_abnormal','noise_1/normal_abnormal_differernt')
    find_logkey_in_train_not_in_test_normal('noise_1/train','noise_1/test_normal','noise_1/not_in_test_normal')
    dir_noise = 'noise_1/noise_0'#
    sim_throad_0 = 0.99
    p_num = 1
    get_importain_word_csv('data/drain_random_with_level_20/BGL.log_templates.csv', 100, 'noise_1/templates_important_word_old')
    get_importain_word_csv('noise_1/BGL.log_templates.csv', 100, 'noise_1/templates_important_word')
    get_map_unkonw_tmp_to_exist('noise_1/templates_important_word_old', 'noise_1/templates_important_word',dir_noise+'/map_know_to_exist',dir_noise+'/save_sim')
    get_map_unkonw_tmp_to_0('noise_1/templates_important_word_old',sim_throad_0,'noise_1/templates_important_word',dir_noise+'/map_know_to_0')
    get_map_0_to_exist(p_num,dir_noise+'/map_know_to_0',dir_noise+'/map_know_to_exist',dir_noise+'/map_know_to_0_new')
    unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', 'noise_1/train',dir_noise + '/WITH_S_train')
    unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', 'noise_1/test_normal',dir_noise + '/WITH_S_test_normal')
    unknown_logkey_to_old_logkey(dir_noise + '/map_know_to_exist', 'noise_1/test_abnormal',dir_noise + '/WITH_S_test_abnormal')
    #find_normal_abnormal_log_different('noise_1/BGL.log_structured.csv','noise_1/logkey_num_map','noise_1/noise_0/WITH_S_train','noise_1/noise_0/WITH_S_test_abnormal','noise_1/noise_0/normal_abnormal_differernt')
    find_logkey_in_train_not_in_test_normal('noise_1/noise_0/WITH_S_train','noise_1/noise_0/WITH_S_test_abnormal','noise_1/noise_0/not_in_test_normal')
    '''
    #find_logkey_normal_abnormal('noise_1/BGL.log_structured.csv','noise_1/logkey_num_map','noise_1/lable_logkey','noise_1/BGL.log_templates.csv')
    #find_logkey_normal_abnormal('struct_data/drain/BGL.log_structured.csv','struct_data/drain/logkey_num_map','noise_1/lable_logkey_row','struct_data/drain/BGL.log_templates.csv')
    #find_error_match('noise_1/lable_logkey','noise_1/lable_logkey_row','noise_1/noise_0/map_know_to_exist','noise_1/noise_0/error')

    #data/drain_dataset_80
    #create_mapfile_logkey('struct_data/BGL.log_templates.csv','struct_data/logkey_num_map') #creat map file such as '070de4aa' map to '1'
    #group_drain(0.8, 'struct_data/BGL.log_structured.csv','struct_data/logkey_num_map' ,'data/drain_dataset_80/train', 'data/drain_dataset_80/test_normal', 'data/drain_dataset_80/test_abnormal')
    #count_logkey_in_trian_test('data/drain_dataset_80/train', 'data/drain_dataset_80/test_abnormal', 'data/drain_dataset_80/test_normal','data/drain_dataset_80/logkey_not_in_train')
    #simple_test_dataset(0.05, 20, 'data/drain_dataset_80/train', 'data/drain_dataset_80/train_simple')
    #simple_test_dataset(0.05, 20, 'data/drain_dataset_80/test_abnormal', 'data/drain_dataset_80/test_abnormal_simple')
    #simple_test_dataset(0.05, 20, 'data/drain_dataset_80/test_normal', 'data/drain_dataset_80/test_normal_simple')
    #find_normal_abnormal_log_different('struct_data/BGL.log_structured.csv','struct_data/logkey_num_map','data/drain_dataset_80/train','data/drain_dataset_80/test_abnormal','data/drain_dataset_80/normal_abnormal_differernt')
    #count_logkey_in_trian_test('data/order_random_dataset_20/train', 'data/order_random_dataset_20/test_normal', 'data/order_random_dataset_20/test_abnormal','data/order_random_dataset_20/logkey_not_in_train')

    #data/drain_with_level_80
    '''
    #create_mapfile_logkey('bgl_dataset/drain_without_06_4/BGL.log_templates.csv','bgl_dataset/drain_without_06_4/logkey_num_map') #creat map file such as '070de4aa' map to '1'
    #group_drain(0.8, 'bgl_dataset/drain_without_06_4/BGL.log_structured.csv','bgl_dataset/drain_without_06_4/logkey_num_map' ,'data/drain_with_level_80/train', 'data/drain_with_level_80/test_normal', 'data/drain_with_level_80/test_abnormal')
    #count_logkey_in_trian_test('data/drain_with_level_80/train', 'data/drain_with_level_80/test_normal', 'data/drain_with_level_80/test_abnormal','data/drain_with_level_80/logkey_not_in_train')
    simple_test_dataset(0.05, 20, 'data/drain_with_level_80/train', 'data/drain_with_level_80/train_simple')
    simple_test_dataset(0.05, 20, 'data/drain_with_level_80/test_abnormal', 'data/drain_with_level_80/test_abnormal_simple')
    simple_test_dataset(0.05, 20, 'data/drain_with_level_80/test_normal', 'data/drain_with_level_80/test_normal_simple')
    #find_normal_abnormal_log_different('bgl_dataset/drain_without_06_4/BGL.log_structured.csv','bgl_dataset/drain_without_06_4/logkey_num_map','data/drain_with_level_80/train','data/drain_with_level_80/test_abnormal','data/drain_with_level_80/normal_abnormal_differernt')
    '''
    #count_logkey_in_trian_test('data/drain_with_level_80/train', 'data/drain_with_level_80/test_normal', 'data/drain_with_level_80/test_abnormal','data/drain_with_level_80/logkey_not_in_train')
    #count_logkey_in_trian_test('data/drain_random_with_level_20/train', 'data/drain_random_with_level_20/test_normal', 'data/drain_random_with_level_20/test_abnormal','data/drain_random_with_level_20/logkey_not_in_train')

    #drain_random_with_level_20
    #loganomaly_dataset(20, 'data/drain_with_level_80/train', 'data/drain_random_with_level_20/hdfs_train')
    #loganomaly_dataset(20,'data/drain_with_level_80/test_normal', 'data/drain_random_with_level_20/hdfs_test_normal')
    #loganomaly_dataset(20, 'data/drain_with_level_80/test_abnormal', 'data/drain_random_with_level_20/hdfs_test_abnormal')
   # random_loganormaly(0.2, 'data/drain_random_with_level_20/hdfs_train', 'data/drain_random_with_level_20/hdfs_test_normal', 'data/drain_random_with_level_20/hdfs_test_abnormal', 'data/drain_random_with_level_20/train_1', 'data/drain_random_with_level_20/test_normal_1', 'data/drain_random_with_level_20/test_abnormal_1')
    #loganormaly_dataset_to_deeplog_dataset('data/drain_random_with_level_20/train_1',\
    #33                                       'data/drain_random_with_level_20/test_normal_1',\
    #                                     'data/drain_random_with_level_20/test_abnormal_1',\
    #                                       'data/drain_random_with_level_20/train',\
    #                                      'data/drain_random_with_level_20/test_normal',\
    #                                       'data/drain_random_with_level_20/test_abnormal')

    #count_logkey_in_trian_test('data/drain_random_with_level_20/train', 'data/drain_random_with_level_20/test_normal', 'data/drain_random_with_level_20/test_abnormal','data/drain_random_with_level_20/logkey_not_in_train')
    #simple_test_dataset(0.05, 20, 'data/drain_random_with_level_20/train', 'data/drain_random_with_level_20/train_simple')
    #simple_test_dataset(0.05, 20, 'data/drain_random_with_level_20/test_abnormal', 'data/drain_random_with_level_20/test_abnormal_simple')
    #simple_test_dataset(0.05, 20, 'data/drain_random_with_level_20/test_normal', 'data/drain_random_with_level_20/test_normal_simple')
    #save_logkey_in_dataset_normal('data/drain_random_with_level_20/train', 'data/drain_random_with_level_20/train_logkey')
    #unknown_logkey_to_0('data/drain_random_with_level_20/train_logkey','data/drain_random_with_level_20/test_abnormal','data/drain_random_with_level_20/test_abnormal_0')
    #unknown_logkey_to_0('data/drain_random_with_level_20/train_logkey','data/drain_random_with_level_20/test_normal','data/drain_random_with_level_20/test_normal_0')
    #data_set_exist_0('data/dataset_80/train','data/dataset_80_exist_0/train')
    #data_set_exist_0('data/dataset_80/test_normal','data/dataset_80_exist_0/test_normal')
    #data_set_exist_0('data/dataset_80/test_abnormal','data/dataset_80_exist_0/test_abnormal')
    #save_logkey_in_dataset_normal('data/dataset_80_exist_0/train', 'data/dataset_80_exist_0/train_logkey')
    #unknown_logkey_to_0('data/dataset_80_exist_0/train_logkey','data/dataset_80_exist_0/test_abnormal','data/dataset_80_exist_0/test_abnormal_new')
    #unknown_logkey_to_0('data/dataset_80_exist_0/train_logkey','data/dataset_80_exist_0/test_normal','data/dataset_80_exist_0/test_normal_new')
    #count_logkey_in_trian_test('data/dataset_80_exist_0/train', 'data/dataset_80_exist_0/test_abnormal', 'data/dataset_80_exist_0/test_normal','data/dataset_80_exist_0/logkey_not_in_train')
    #csvtemplates_to_templates('bgl_dataset/drain_without_06_4/logkey_num_map','bgl_dataset/drain_without_06_4/BGL.log_templates.csv','data/drain_random_with_level_20/templates')
    #get_importain_word_Tf_idf('data/word2vec_model_100d', 'data/drain_random_with_level_20/templates', 'data/glove.6B.100d.word2vec.txt', 4,
    #                          'data/drain_random_with_level_20/event_vector_top4.txt','data/drain_random_with_level_20/save.txt')
    #loganomaly_dataset(20, 'data/drain_random_with_level_20/train', 'data/drain_random_with_level_20/hdfs_train')
    #loganomaly_dataset(20,'data/drain_random_with_level_20/test_normal', 'data/drain_random_with_level_20/hdfs_test_normal')
    #loganomaly_dataset(20, 'data/drain_random_with_level_20/test_abnormal', 'data/drain_random_with_level_20/hdfs_test_abnormal')
    ##get_importain_word_Tf_idf('data/word2vec_model_100d', 'data/drain_random_with_level_20/templates', 'data/glove.6B.100d.word2vec.txt',3 ,
     #                         'data/drain_random_with_level_20/event_vector_top3.txt','data/drain_random_with_level_20/save_top3.txt')
    #get_importain_word_Tf_idf_with_level('data/word2vec_model_100d', 'data/drain_random_with_level_20/templates', 'data/glove.6B.100d.word2vec.txt',2 ,
    #                           'data/drain_random_with_level_20/event_vector_top2_with_level.txt','data/drain_random_with_level_20/save_top3_with_levels.txt')
    #order_random_dataset_20_loganomaly_exist_0
    '''
    loganomaly_dataset(20, 'data/dataset_80_exist_0/train', 'data/log_anomaly_exist_0/hdfs_train')
    loganomaly_dataset(20,'data/dataset_80_exist_0/test_normal', 'data/log_anomaly_exist_0/hdfs_test_normal')
    loganomaly_dataset(20, 'data/dataset_80_exist_0/test_abnormal', 'data/log_anomaly_exist_0/hdfs_test_abnormal')
    random_loganormaly(0.2, 'data/log_anomaly_exist_0/hdfs_train', 'data/log_anomaly_exist_0/hdfs_test_normal', 'data/log_anomaly_exist_0/hdfs_test_abnormal', 'data/order_random_dataset_20_loganomaly_exist_0/train', 'data/order_random_dataset_20_loganomaly_exist_0/test_normal', 'data/order_random_dataset_20_loganomaly_exist_0/test_abnormal')
    loganormaly_dataset_to_deeplog_dataset('data/order_random_dataset_20_loganomaly_exist_0/train',\
                                           'data/order_random_dataset_20_loganomaly_exist_0/test_normal',\
                                           'data/order_random_dataset_20_loganomaly_exist_0/test_abnormal',\
                                           'data/order_random_dataset_20_exist_0/train',\
                                          'data/order_random_dataset_20_exist_0/test_normal',\
                                           'data/order_random_dataset_20_exist_0/test_abnormal')

    count_logkey_in_trian_test('data/order_random_dataset_20_exist_0/train', 'data/order_random_dataset_20_exist_0/test_abnormal', 'data/order_random_dataset_20_exist_0/test_normal','data/order_random_dataset_20_exist_0/logkey_not_in_train')
    save_logkey_in_dataset_normal('data/order_random_dataset_20_exist_0/train', 'data/order_random_dataset_20_exist_0/train_logkey')
    unknown_logkey_to_0('data/order_random_dataset_20_exist_0/train_logkey','data/order_random_dataset_20_exist_0/test_abnormal','data/order_random_dataset_20_exist_0/test_abnormal_new')
    unknown_logkey_to_0('data/order_random_dataset_20_exist_0/train_logkey','data/order_random_dataset_20_exist_0/test_normal','data/order_random_dataset_20_exist_0/test_normal_new')
    '''
    #random_loganormaly(0.2, 'data/log_anomaly/hdfs_train', 'data/log_anomaly/hdfs_test_normal', 'data/log_anomaly/hdfs_test_abnormal', 'data/order_random_dataset_20_loganomaly/train', 'data/order_random_dataset_20_loganomaly/test_normal', 'data/order_random_dataset_20_loganomaly/test_abnormal')
    #count_logkey_in_trian_test('data/order_random_dataset_20/train', 'data/order_random_dataset_20/test_abnormal', 'data/order_random_dataset_20/test_normal','data/order_random_dataset_20/logkey_not_in_train')
    #save_logkey_in_dataset_normal('data/order_random_dataset_20/train', 'data/order_random_dataset_20/train_logkey')
    #unknown_logkey_to_0('data/order_random_dataset_20/train_logkey','data/order_random_dataset_20/test_abnormal','data/order_random_dataset_20/test_abnormal_new')
    #unknown_logkey_to_0('data/order_random_dataset_20/train_logkey','data/order_random_dataset_20/test_normal','data/order_random_dataset_20/test_normal_new')
    #loganormaly_dataset_to_deeplog_dataset('data/order_random_dataset_20_loganomaly/train',\
    #                                       'data/order_random_dataset_20_loganomaly/test_normal',\
    #                                       'data/order_random_dataset_20_loganomaly/test_abnormal',\
    #                                       'data/order_random_dataset_20/train',\
    #                                      'data/order_random_dataset_20/test_normal',\
    #                                       'data/order_random_dataset_20/test_abnormal')
    #random_split_dataset(0.2, 'data/normal_new', 'data/abnormal_new', 'data/random_dataset_20/train', 'data/random_dataset_20/test_normal', 'data/random_dataset_20/test_abnormal')
    #count_logkey_in_trian_test('data/random_dataset_20/train', 'data/random_dataset_20/test_abnormal', 'data/random_dataset_20/test_normal','data/random_dataset_20/logkey_not_in_train')
    #splite_top_dataset(0.8, 'data/normal_new', 'data/abnormal_new', 'data/dataset_80_window_20/train', 'data/dataset_80_window_20/test_normal', 'data/dataset_80_window_20/test_abnormal')
    #loganomaly_dataset(20, 'data/dataset_80/train', 'data/log_anomaly/hdfs_train')
    #loganomaly_dataset( 20,'data/dataset_80/test_normal', 'data/log_anomaly/hdfs_test_normal')
    #loganomaly_dataset(20, 'data/dataset_80/test_abnormal', 'data/log_anomaly/hdfs_test_abnormal')
    #save_logkey_in_dataset_normal('data/dataset_80/train', 'data/dataset_80/train_logkey')
    #unknown_logkey_to_0('data/dataset_80/train_logkey','data/dataset_80/test_abnormal','data/dataset_80/test_abnormal_new')
    #unknown_logkey_to_0('data/dataset_80/train_logkey','data/dataset_80/test_normal','data/dataset_80/test_normal_new')
    #save_logkey_in_dataset_test('data/dataset_80/test_normal', 'data/dataset_80/test_abnormal', 'data/dataset_80/test_logkey')
    #count_logkey_in_trian_test('data/dataset_80/train','data/dataset_80/test_abnormal','data/dataset_80/test_normal')

    #2020.12.21
    #split_dataset('dataset_1',0.01)
    #count_logkey('data/dataset_1/train')
    #split_data('data/BGL.log',5 ,'data/BGL_800k.log')
    #split_data_top('data/BGL.log',10,'data/BGL_top_10.log')
    #word2vec_train('data/BGL_top_10.log', 'data/word2vec_model_100d','data/word2vec_model_100d_model')
    #get_full_wordvec('data/templates','data/word2vec_model_100d')
    #get_importain_word_Tf_idf('data/word2vec_model_100d','data/templates','data/glove.6B.100d.word2vec.txt',4,'data/event_vector_top3.txt')
    #save_logkey_in_dataset('data/dataset_10/train','data/dataset_10/train_logkey')
    #save_logkey_in_dataset('data/dataset_10/test','data/dataset_10/abnormal','data/dataset_10/test_logkey')
    #map_testdataset_logkey_to_traindataset_logkey('data/dataset_10/train_logkey','data/dataset_10/test_logkey','data/event_vector_top3.txt','data/dataset_10/map')
    #unknown_logkey_to_old_logkey('data/dataset_10/map','data/dataset_10/test','data/dataset_10/test_new')
    #unknown_logkey_to_old_logkey('data/dataset_10/map','data/dataset_10/abnormal','data/dataset_10/abnormal_new')
    #deeplog
    #unknown_logkey_to_0('data/dataset_10/train_logkey','data/dataset_10/abnormal','data/dataset_10/abnormal_deeplog')
    #unknown_logkey_to_0('data/dataset_10/train_logkey','data/dataset_10/test','data/dataset_10/test_deeplog')
    #dimantion_of_logket2vec('data/event_vector_top3.txt')

    #dataset 1%
    #save_logkey_in_dataset('data/dataset_1/train','data/dataset_1/train_logkey')
    #save_logkey_in_dataset('data/dataset_1/test', 'data/dataset_1/abnormal', 'data/dataset_1/test_logkey')
    #map_testdataset_logkey_to_traindataset_logkey('data/dataset_1/train_logkey','data/dataset_1/test_logkey','data/event_vector_top3.txt','data/dataset_1/map')
    #unknown_logkey_to_old_logkey('data/dataset_1/map','data/dataset_1/test','data/dataset_1/test_new')
    #unknown_logkey_to_old_logkey('data/dataset_1/map','data/dataset_1/abnormal','data/dataset_1/abnormal_new')
    #deeplog
    #unknown_logkey_to_0('data/dataset_1/train_logkey','data/dataset_1/abnormal','data/dataset_1/abnormal_deeplog')
    #unknown_logkey_to_0('data/dataset_1/train_logkey','data/dataset_1/test','data/dataset_1/test_deeplog')

    #drain structure data to logkey with 1% dataset for training
    #divide_normal_abnormal('struct_data/BGL.log_structured.csv','struct_data/drain_dataset_1/normal','struct_data/drain_dataset_1/abnormal')    #divide drain structure dataset to normal and abmormal dataset and those dataset only contain logkey
    #create_mapfile_logkey('struct_data/BGL.log_templates.csv','struct_data/logkey_num_map') #creat map file such as '070de4aa' map to '1'
    #map_logkey_to_num('struct_data/logkey_num_map','struct_data/drain_dataset_1/normal','struct_data/drain_dataset_1/normal_num') #map drain logkey to number such as '070de4aa' to '1', and
    #map_logkey_to_num('struct_data/logkey_num_map','struct_data/drain_dataset_1/abnormal','struct_data/drain_dataset_1/abnormal_num') #map drain logkey to number such as '070de4aa' to '1', and
    #split_dataset('struct_data/drain_dataset_1/normal_num', 0.01, 'struct_data/drain_dataset_1/train','struct_data/drain_dataset_1/normal_test')
    #count_logkey('struct_data/drain_dataset_1/train')
    #count_logkey('struct_data/drain_dataset_1/normal_test')
    #count_logkey('struct_data/drain_dataset_1/abnormal_num')
    #csvtemplates_to_templates('struct_data/logkey_num_map','struct_data/BGL.log_templates.csv','struct_data/templates')
    #get_importain_word_Tf_idf('data/word2vec_model_100d','struct_data/templates','data/glove.6B.100d.word2vec.txt',4,'struct_data/event_vector_top3.txt','struct_data/logkey2vec_4_save')
    #save_logkey_in_dataset_normal('struct_data/drain_dataset_1/train','struct_data/drain_dataset_1/train_logkey')
    #save_logkey_in_dataset_test('struct_data/drain_dataset_1/normal_test','struct_data/drain_dataset_1/abnormal_num','struct_data/drain_dataset_1/test_logkey')
    #our method
    #map_testdataset_logkey_to_traindataset_logkey('struct_data/drain_dataset_1/train_logkey','struct_data/drain_dataset_1/test_logkey','struct_data/event_vector_top3.txt','struct_data/drain_dataset_1/map')
    #unknown_logkey_to_old_logkey('struct_data/drain_dataset_1/map','struct_data/drain_dataset_1/normal_test','struct_data/drain_dataset_1/dataset_test_normal')
    #unknown_logkey_to_old_logkey('struct_data/drain_dataset_1/map','struct_data/drain_dataset_1/abnormal_num','struct_data/drain_dataset_1/dataset_test_abnormal')
    #deeplog
    #unknown_logkey_to_0('struct_data/drain_dataset_1/train_logkey','struct_data/drain_dataset_1/abnormal_num','struct_data/drain_dataset_1/dataset_test_abnormal_deeplog')
    #unknown_logkey_to_0('struct_data/drain_dataset_1/train_logkey','struct_data/drain_dataset_1/normal_test','struct_data/drain_dataset_1/dataset_test_normal_deeplog')
    #dimantion_of_logket2vec('data/event_vector_top3.txt')

    #drain structure data to logkey with 10% dataset for trainingpy
    #split_dataset('struct_data/drain_dataset_1/normal_num', 0.1, 'struct_data/drain_dataset_10/train','struct_data/drain_dataset_10/normal_test')
    #count_logkey('struct_data/drain_dataset_10/train')
    #count_logkey('struct_data/drain_dataset_10/normal_test')
    #count_logkey('struct_data/drain_dataset_10/abnormal_num')
    #save_logkey_in_dataset_normal('struct_data/drain_dataset_10/train','struct_data/drain_dataset_10/train_logkey')
    #save_logkey_in_dataset_test('struct_data/drain_dataset_10/normal_test','struct_data/drain_dataset_10/abnormal_num','struct_data/drain_dataset_10/test_logkey')
    #our method
    #map_testdataset_logkey_to_traindataset_logkey('struct_data/drain_dataset_10/train_logkey','struct_data/drain_dataset_10/test_logkey','struct_data/event_vector_top3.txt','struct_data/drain_dataset_10/map')
    #unknown_logkey_to_old_logkey('struct_data/drain_dataset_10/map','struct_data/drain_dataset_10/normal_test','struct_data/drain_dataset_10/dataset_test_normal')
    #unknown_logkey_to_old_logkey('struct_data/drain_dataset_10/map','struct_data/drain_dataset_10/abnormal_num','struct_data/drain_dataset_10/dataset_test_abnormal')
    #deeplog
    #unknown_logkey_to_0('struct_data/drain_dataset_10/train_logkey','struct_data/drain_dataset_10/abnormal_num','struct_data/drain_dataset_10/dataset_test_abnormal_deeplog')
    #unknown_logkey_to_0('struct_data/drain_dataset_10/train_logkey','struct_data/drain_dataset_10/normal_test','struct_data/drain_dataset_10/dataset_test_normal_deeplog')