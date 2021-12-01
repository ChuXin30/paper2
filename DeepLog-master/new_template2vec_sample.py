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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

event_output_file = 'new_template2vec/event_vector_top3.txt'
event_model = KeyedVectors.load_word2vec_format(event_output_file, binary=False)

#word2vec_output_file = 'new_template2vec/word2vec_100d.txt'
#word2vec_output_file = 'new_template2vec/word2vec.txt'
#word2vec_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

#word2vec_output_file_1 = 'new_template2vec/word2vec_100d.txt'
#word2vec_model_1 = KeyedVectors.load_word2vec_format(word2vec_output_file_1, binary=False)

#glove_file = 'new_template2vec/glove.6B.100d.word2vec.txt'
#glove_file_model = KeyedVectors.load_word2vec_format(glove_file,binary=False)

def word2vec_train():
    num_features = 100  # Word vector dimensionality
    min_word_count = 1  # Minimum word count
    num_workers = 16  # Number of threads to run in parallel
    context = 4  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    dict_train = {}
    f_train = open('data_word/HDFS_group_normal.txt', mode='r')
    for line in f_train:
        list_line = line.split(" ")
        for var in list_line:
            var = var.strip('\n')
            if var not in dict_train:
                dict_train[var] = 1
            else:
                dict_train[var] += 1
    f_train = open('data_word/HDFS_group_abnormal.txt', mode='r')
    for line in f_train:
        list_line = line.split(" ")
        for var in list_line:
            var = var.strip('\n')
            if var not in dict_train:
                dict_train[var] = 1
            else:
                dict_train[var] += 1

    #    print(sorted(dict_train))
    #    print(len(dict_train))
    #    print(dict_train)

    #    print(glove_model.wv.vocab.keys())
    #    print(sorted(glove_model.wv.vocab.keys()))
    #    print(len(glove_model.wv.vocab.keys()))
    # -1
    #    dict_importain_word = { 'Received': 0, 'block': 13,  'of': 2, 'size': 4, 'Deleting': 0, 'file': 2, 'BLOCK*': 6, 'NameSystem.allocateBlock:': 0, 'to': 13, 'Receiving': 1, 'for': 8, 'PacketResponder': 1, 'terminating': 0, 'src:': 0, 'dest:': 0, 'Starting': 0, 'thread': 0, 'transfer': 0, 'NameSystem.delete:': 0, 'is': 2, 'added': 1, 'invalidSet': 0, 'NameSystem.addStoredBlock:': 3, 'blockMap': 0, 'updated:': 0, 'Unexpected': 0, 'error': 0, 'trying': 0, 'delete': 0, 'BlockInfo': 0, 'not': 1, 'found': 0, 'in': 1, 'volumeMap.': 0, 'exception': 4, 'while': 2, 'serving': 0, 'ask': 0, 'replicate': 0, 'datanode(s)': 0, 'writeBlock': 3, 'received': 6, 'addStoredBlock': 2, 'request': 2, 'on': 3, 'java.io.IOException:': 2,  'Exception': 2, 'receiveBlock': 0, 'Changing': 0, 'offset': 1, 'from': 0, '0': 0, 'meta': 0, 'java.io.InterruptedIOException:': 0, 'Interruped': 0, 'waiting': 1, 'IO': 0, 'channel': 1, 'millis': 1, 'timeout': 1, 'left.': 0, '1': 0, 'But': 0, 'it': 0, 'does': 0, 'belong': 0, 'any': 0, 'file.': 0, 'PendingReplicationMonitor': 0, 'timed': 0, 'out': 0,  'writing': 0, 'mirror': 0,  'java.net.SocketTimeoutException:': 0,  'be': 1, 'ready': 0, 'write.': 0, 'ch': 0, ':': 0, 'java.nio.channels.SocketChannel': 0, 'connected': 0,  'Connection': 0, 'reset': 0, 'by': 0, 'peer': 0, 'Block': 0, 'valid': 0, 'and': 0, 'cannot': 0, 'written': 0, 'to.': 0, 'Adding': 0, 'an': 0, 'already': 0, 'existing': 0}
    # -1.5
    #    dict_importain_word= { 'block': 10, 'BLOCK*': 3, 'NameSystem.allocateBlock:': 0, 'to': 10, 'Receiving': 0, 'for': 6, 'Starting': 0, 'thread': 0, 'transfer': 0, 'is': 1, 'added': 0, 'Received': 0, 'of': 1, 'size': 1, 'Unexpected': 0, 'error': 0, 'trying': 0, 'delete': 0, 'BlockInfo': 0, 'not': 0, 'found': 0, 'in': 1, 'volumeMap.': 0, 'exception': 4, 'while': 2, 'serving': 0, 'ask': 0, 'replicate': 0, 'datanode(s)': 0, 'writeBlock': 3, 'received': 4, 'NameSystem.addStoredBlock:': 0, 'addStoredBlock': 0, 'request': 0, 'on': 1, 'java.io.IOException:': 2, 'Changing': 0, 'file': 1, 'offset': 1, 'from': 0, '0': 0, 'meta': 0, 'Exception': 2, 'java.io.InterruptedIOException:': 0, 'Interruped': 0, 'waiting': 1, 'IO': 0, 'channel': 1, 'millis': 1, 'timeout': 1, 'left.': 0, 'PacketResponder': 0, '1': 0, 'receiveBlock': 0, 'java.net.SocketTimeoutException:': 0,  'be': 1, 'ready': 0, 'write.': 0, 'ch': 0, ':': 0, 'java.nio.channels.SocketChannel': 0, 'connected': 0, 'Connection': 0, 'reset': 0, 'by': 0, 'peer': 0, 'Block': 0, 'valid': 0, 'and': 0, 'cannot': 0, 'written': 0, 'to.': 0, 'Adding': 0, 'an': 0, 'already': 0, 'existing': 0}
    # -3
    #    dict_importain_word= { 'block': 11557, 'BLOCK*': 6, 'NameSystem.allocateBlock:': 0, 'for': 1, 'to': 121, 'from': 5, 'size': 52, '67108864': 45, 'exception': 4, 'Exception': 0, 'java.io.InterruptedIOException:': 0, 'Interruped': 0, 'while': 1, 'waiting': 0, 'IO': 0, 'on': 0, 'channel': 0, 'millis': 0, 'timeout': 0, 'left.': 0, 'PacketResponder': 0, '1': 0, 'received': 1, 'not': 0, 'java.io.IOException:': 1, 'Connection': 0, 'reset': 0, 'by': 0, 'peer': 0, 'serving': 0, 'is': 0, 'added': 0, 'writeBlock': 0}
    fr = open("data_word/HDFS_train.txt", mode='r')
    #    sentences = word2vec.Text8Corpus("data/HDFS_train.txt")
    sentences = word2vec.Text8Corpus("data_word/HDFS_train.txt")

    # print(len(sentences))

    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                              size=num_features, min_count=min_word_count, \
                              window=context, sg=1, sample=downsampling)
    model.init_sims(replace=True)
    # 保存模型，供日後使用
    model.save("data_word/word2vec_model_100d")
    f = open('data_word/word2vec_100d.txt', mode='w')

    #    print(model)
    print(model)
    for key in model.wv.vocab.keys():
        #    print(key)
        #    print(model[key])

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

def get_importain_word(name,imp,top_k,save_file):
    fr = open(name,mode='r')
    corpus = []

    #
    for line in fr:
        line = line.replace('\n',' ')
        r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = re.sub(r1, ' ' , line).lower()
        line2 = ''
        for var in line.split(' '):
            line2 += ' '+var
       # r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
       # line = re.sub(r1, ' ' , )
       # key = get_blk_id(line)
       # line = line.replace(key,'')
        corpus.append(line2)
    #将文本中的词语转换为词频矩阵

    vectorizer = CountVectorizer(lowercase=False)
    #计算个词语出现的次数
    X = vectorizer.fit_transform(corpus)
 #   print(X)
    #获取词袋中所有文本关键词
    word = vectorizer.get_feature_names()
    print(word)


    word2vec_output_file = 'new_template2vec/word2vec_full.txt'
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


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

   # top_k = 3
    fw = open(save_file,mode='w')
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
            if word[j] not in word2vec_model:
                print("word not in word2vec model {}".format(word[j]))

            x = word2vec_model[word[j]].tolist()
            list_x = []
            for var in x:
                list_x.append(str(var))
            list_x = " ".join(list_x)
            event_vector += list_x
            if count < top_k-1:
                #print(j)
                event_vector += ' '
            count += 1
        fw.write(str(i) + ' ' + event_vector + '\n')
        print(str(i)+' '+s+' '+s_w)



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

def create_word_in_col_head():
    f = open('new_template2vec/col_header.txt',mode='r')
    fw = open('new_template2vec/word2vec_full.txt',mode='w')
    dict_word =  {}
    for line in f:
        r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = line.replace('\n',' ')
        line = re.sub(r1, ' ' , line)
        for var in line.split(' '):
            var = var.lower()
            if var not in dict_word:
                dict_word[var] = 1
    print(len(dict_word))
    print(dict_word)

    for word in dict_word:
        if word in glove_file_model:
            x = glove_file_model[word].tolist()
            list_x = []
            for var in x:
                list_x.append(str(var))
            list_x = " ".join(list_x)
            fw.write(word + ' ' + list_x + '\n')
        elif word in word2vec_model_1:
            x = word2vec_model_1[word].tolist()
            list_x = []
            for var in x:
                list_x.append(str(var))
            list_x = " ".join(list_x)
            fw.write(word + ' ' + list_x + '\n')
        elif word in word2vec_model:
            x = word2vec_model[word].tolist()
            list_x = []
            for var in x:
                list_x.append(str(var))
            list_x = " ".join(list_x)
            fw.write(word + ' ' + list_x + '\n')

def word2vec_full_in_colhead():
    word2vec_output_file = 'new_template2vec/word2vec_full.txt'
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    f = open('new_template2vec/col_header.txt',mode='r')
    dict_word =  {}
    for line in f:
        r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = line.replace('\n',' ')
        line = re.sub(r1, ' ' , line).lower()
        print(line)
        for var in line.split(' '):
            if var not in dict_word:
                dict_word[var] = 1
    print(len(dict_word))
    print(dict_word)

    for word in dict_word:
        if word not in word2vec_model:
            print(word)

def get_importain_word_Tf_idf(name,imp,top_k,save_file):
    fr = open(name,mode='r')
    corpus = []

    #
    for line in fr:
        line = line.replace('\n',' ')
        r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = re.sub(r1, ' ' , line).lower()
        line2 = ''
        for var in line.split(' '):
            line2 += ' '+var
       # r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
       # line = re.sub(r1, ' ' , )
       # key = get_blk_id(line)
       # line = line.replace(key,'')
        corpus.append(line2)
    #将文本中的词语转换为词频矩阵

    vectorizer = CountVectorizer(lowercase=False)
    #计算个词语出现的次数
    X = vectorizer.fit_transform(corpus)
 #   print(X)
    #获取词袋中所有文本关键词
    word = vectorizer.get_feature_names()
    print(word)
    word2vec_output_file = 'new_template2vec/word2vec_full.txt'
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
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

   # top_k = 3
    fw = open(save_file,mode='w')
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
        '''
            if word[j] not in word2vec_model:
                print("word not in word2vec model {}".format(word[j]))
            x = word2vec_model[word[j]].tolist()
           # x = word[j]
            print(x)
            print(type(x))
            print(type(x[0]))
            print(weight[i][j])
            print(type(float(weight[i][j])))
            #for i in range(len(x)):
            #    print(x[i])
            x = [x[k]* float(weight[i][j]) for k in range(len(x))] # mutiply tf-idf
            list_x = []
            for var in x:
                list_x.append(str(var))
            list_x = " ".join(list_x)
            event_vector += list_x
            if count < top_k-1:
                #print(j)
                event_vector += ' '
            count += 1
        fw.write(str(i) + ' ' + event_vector + '\n')
        '''
        print(str(i)+' '+s+' '+s_w)

if __name__ == "__main__":
    #create_word_in_col_head()
    #word2vec_full_in_colhead()
    #get_importain_word('new_template2vec/col_header.txt',0,4,'new_template2vec/event_vector_top4.txt')
    #get_importain_word('new_template2vec/col_header.txt',0,2,'new_template2vec/event_vector_top2.txt')
    #get_importain_word('new_template2vec/col_header.txt',0,6,'new_template2vec/event_vector_top6.txt')
#    get_importain_word('new_template2vec/col_header.txt',0,1,'new_template2vec/event_vector_top1.txt')
    #get_importain_word('new_template2vec/col_header.txt',0,3,'new_template2vec/event_vector_top3.txt')

    #get_importain_word_Tf_idf('new_template2vec/col_header.txt',0,4,'new_template2vec/event_vector_top4_tfidf.txt')
    get_importain_word_Tf_idf('new_template2vec/templates',0,8,'new_template2vec/templates.txt')

#    print(event_model['0'])
#    print(len(event_model['0']))
    #fr  = open('new_template2vec/event_vector_top3.txt',mode='r')
    #for line in fr:
    #    print(len(line.split(' ')))