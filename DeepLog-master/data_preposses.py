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
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 
import datetime

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
#    print(dict_blk_id['blk_-1067131609371010449'])
#    print(dict_blk_id['blk_-1608999687919862906'])    
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
        dict_blk_id[key] = dict_blk_id[key].replace('\n','')
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
        line = line.replace('\n',' ')
        #line = line.replace('\n',' ').lower()
        line = re.sub(r1, ' ' , line)
        for var in line.split(' '):
            if var not in dict_word:
                dict_word[var] = 1
                #fw.write(var+'\n')
            else:
                dict_word[var] += 1
    print(dict_word)
    print(len(dict_word))
    dict_important_word = {}
    for var in dict_word:
        if dict_word[var] == 1 or dict_word[var] == 1:
            fw.write(var+'\n')
            dict_important_word[var] = 1

    dict_sort = list(sorted(dict_word.items(),key=lambda d:d[1] ,reverse=False ))

    print(dict_sort)
    f = open('col_header.txt',mode='r')    
    for line in f :
        r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = line.replace('\n',' ')
        #line = line.replace('\n',' ').lower()
        line = re.sub(r1, ' ' , line)   
        s = ""
        for var in line.split():
            if var in dict_important_word:
                s += ' ' +var
        print(s)
    '''
    list_f = []
    for line in f :
        r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = line.replace('\n',' ').lower()
        line = re.sub(r1, ' ' , line)   
        list_f.append(line)

    print(list_f)
    for word in dict_sort:
        for line in list_f:
            list_line = line.split(' ')
            if word[0] in list_line:
#                dict_sort.remove(word)
                print(word)
                dict_sort.remove(word)
                list_f.remove(line)
            
    print(dict_sort)
#        print(line_important)
    '''

def word_to_key(name):
    f_dict_word = open('word_in_clo_header.txt',mode='r')
    dict_word = {}
    count = 0
    for var in f_dict_word:
        var =var.replace('\n','').lower()
        dict_word[var] = count 
        count += 1
    #print(dict_word)
    #print(len(dict_word))
    #print(sorted(dict_word))
    fr = open('data_word/'+name,mode='r')
    fw = open('wordkey_data/'+name,mode='w')

    for line in fr:
        line_var = ''

        for var in line.split(' '):
            var = var.replace('\n','')
            if var in dict_word:
                line_var += ' ' + str(dict_word[var])
                if var not in dict_in_dataset:
                    dict_in_dataset[var] = 1
                else:
                    dict_in_dataset[var] += 1
        fw.write(line_var.strip()+'\n')
    print(dict_in_dataset)
    print(len(dict_in_dataset))

def get_importain_word(name,imp):
    fr = open(name,mode='r')
    corpus = []

    for line in fr:
        line = line.replace('\n',' ')
        r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = line.replace('\n',' ').lower()
        line = re.sub(r1, ' ' , line)        
        line2 = ''
        for var in line.split(' '):
            if '.' not in var and '\\' not in var:
                line2 += ' '+var
        corpus.append(line2)

    vectorizer = CountVectorizer(lowercase=False)  
    X = vectorizer.fit_transform(corpus)  

    word = vectorizer.get_feature_names()     
    print(word)  
    transformer = TfidfTransformer()  
    tfidf = transformer.fit_transform(X)  
    weight = tfidf.toarray()
    print(len(word))
    print(weight[1])
    importain_word = {}

    for i in range(len(weight)):
        for j in range(len(word)):
            print(weight[i][j])
            if weight[i][j] > imp:
                if word[j] not in importain_word:
                    importain_word[word[j]] = 1
                else:
                    importain_word[word[j]] += 1
    print(len(importain_word))
    print(importain_word)
    print(len(importain_word))
    print(imp)

def logkey_count():
    fr_train = open('wordkey_data/hdfs_train',mode='r')
    fr_test_normal = open('wordkey_data/hdfs_test_normal',mode='r')
    fr_test_abnormal = open('wordkey_data/hdfs_test_abnormal',mode='r')

    dict_in_dataset = {}
    for line in fr_train:
        line_var = ''
        for var in line.split(' '):
            var = var.replace('\n','')
            if var not in dict_in_dataset:
                dict_in_dataset[var] = 1
            else:
                dict_in_dataset[var] += 1
    print(len(dict_in_dataset))
    print(dict_in_dataset)
    for line in fr_test_normal:
        line_var = ''
        for var in line.split(' '):
            var = var.replace('\n','')
            if var not in dict_in_dataset:
                dict_in_dataset[var] = 1
            else:
                dict_in_dataset[var] += 1
    print(len(dict_in_dataset))
    print(dict_in_dataset)

    for line in fr_test_abnormal:
        line_var = ''
        for var in line.split(' '):
            var = var.replace('\n','')
            if var not in dict_in_dataset:
                dict_in_dataset[var] = 1
            else:
                dict_in_dataset[var] += 1
    print(len(dict_in_dataset))
    print(dict_in_dataset)

if __name__ == "__main__":


    '''
    starttime = datetime.datetime.now()

    group_blk_id()
    word_to_key('hdfs_train')
    word_to_key('hdfs_test_normal')
    word_to_key('hdfs_test_abnormal')

    #logkey_count()

    endtime = datetime.datetime.now()
    print((endtime - starttime))
    '''
#    create_word_in_col_head()
#    get_importain_word('word_in_clo_header.txt',0.1)