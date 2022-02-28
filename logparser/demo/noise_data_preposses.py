# -*- coding: utf-8 -*
import pandas as pd
# from gensim.scripts.glove2word2vec import glove2word2vec
# from gensim.models import KeyedVectors
import re
# from sklearn.model_selection import train_test_split
import numpy as np
# from tflearn.data_utils import to_categorical, pad_sequences
import os
# import gensim
import numpy as np
# from gensim.models import word2vec
# from gensim.models import word2vec
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
import datetime
import  random
import csv
from scipy.spatial.distance import cosine
import  random
from sklearn.utils import shuffle
import sys


def get_blk_id(line):
    # 'blk_-1608999687919862906'
    # s = '081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906\n src: /10.250.19.102:54106 dest: /10.250.19.102:50010\n'
    obj = re.search(r" blk_([0-9]*?)\. (.*?)", line)
    blk_id = '1'
    if obj != None:
        #   print('1')
        # print(obj.group(1))
        blk_id = obj.group(1)
    else:
        obj2 = re.search(r" blk_([0-9]*?)\n", line)
#        obj2 = re.search(r" blk_([0-9]*?)", line)
        if obj2 != None:
            #      print('2')
            # print(obj.group(1))
            blk_id = obj2.group(1)
        else:
            obj3 = re.search(r" blk_([0-9]*?) (.*?)", line)
            if obj3 != None:
                #         print('99999')
                blk_id = obj3.group(1)

    if blk_id != '1':
        blk_id = 'blk_' + blk_id
        return blk_id, line

    obj = re.search(r" blk_-([0-9]*?)\. (.*?)", line)
    blk_id = '1'
    if obj != None:
        #   print('1')
        # print(obj.group(1))
        blk_id = obj.group(1)
    else:
#        obj2 = re.search(r" blk_-([0-9]*?)\n", line)
        obj2 = re.search(r" blk_-([0-9]*?)\n", line)
        if obj2 != None:
            #      print('2')
            # print(obj.group(1))
            blk_id = obj2.group(1)
        else:
            obj3 = re.search(r" blk_-([0-9]*?) (.*?)", line)
            if obj3 != None:
                #         print('99999')
                blk_id = obj3.group(1)

    if blk_id != '1':
        blk_id = 'blk_-' + blk_id

    # print(blk_id)
    return blk_id, line

def sub_head(line):  # 去除头部时间戳、日志ｉｄ　只保留日志的信息
    line_without_head = ''
    line_list = line.split(" ")[5:]
    line_without_head = ' '.join(line_list)
    return line_without_head

def group_blk_id():
    #print(word2vec)
    # f = open('data/HDFS.log',mode='r')
    # csv_lable = pd.read_csv('data/anomaly_label.csv',nrows =num)
    csv_lable = pd.read_csv('data/anomaly_label.csv')
    #print(csv_lable)
    csv_lable = csv_lable.values.tolist()

    dict_blk_id = {}
    dict_blk_id_label = {}
    for csv_lable_line in csv_lable:
        dict_blk_id[csv_lable_line[0]] = ''
        dict_blk_id_label[csv_lable_line[0]] = csv_lable_line[1]

    f_dict_word = open('word_in_clo_header.txt', mode='r')
    dict_word = {}
    for var in f_dict_word:
        var = var.replace('\n', '').lower()
        dict_word[var] = 0
    #print(dict_word)
    #print(len(dict_word))

    f = open('data/HDFS.log', mode='r')  # 将数据按照blk_id进行group
    j = 0
    for line in f.readlines():
        if j % 100000 == 0:
            print(j)
        blk_id, s = get_blk_id(line)
        line = sub_head(line)
        # print(blk_id)
        if blk_id in dict_blk_id:
            dict_blk_id[blk_id] = dict_blk_id[blk_id] + line
        else:
            print(blk_id)
            print(s)
        j = j + 1
    # print(dict_blk_id['blk_-1608999687919862906'])
    #
    #    print(dict_blk_id['blk_-1067131609371010449'])
    #    print(dict_blk_id['blk_-1608999687919862906'])
    #    r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    #    r1 = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    j = 0
    list_normal = []
    list_normal_lable = []

    f_dict_abnormal = open('data_word/hdfs_test_abnormal', mode='w')

    for key in dict_blk_id:
        line_blk_vac = ''
        #        dict_blk_id[key] = sub_num_ip_filename(dict_blk_id[key],key)# 去除ｉｐ地址和 blk_id
        #        dict_blk_id[key] = re.sub(r1, ' ', dict_blk_id[key])
        r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        dict_blk_id[key] = re.sub(r1, ' ', dict_blk_id[key])
        dict_blk_id[key] = dict_blk_id[key].lower()  # blk_-1067131609371010449
        dict_blk_id[key] = dict_blk_id[key].replace('\n', '')
        for var in dict_blk_id[key].split(' '):
            if var in dict_word:
                line_blk_vac += ' ' + var
        if dict_blk_id_label[key] == 'Normal':
            list_normal.append(line_blk_vac)
            list_normal_lable.append(0)
        else:
            f_dict_abnormal.write(line_blk_vac + '\n')

        if j % 10000 == 0:
            print(j)
        j = j + 1
    f_dict_test_normal = open('data_word/hdfs_test_normal', mode='w')
    f_dict_train_normal = open('data_word/hdfs_train', mode='w')

    x_test_normal, x_train_normal, y_train, y_test = train_test_split(list_normal, list_normal_lable, test_size=0.01,
                                                                      random_state=0)

    for line in x_test_normal:
        f_dict_test_normal.write(line + '\n')
    for line in x_train_normal:
        f_dict_train_normal.write(line + '\n')

    f_dict_test_normal.close()
    f_dict_train_normal.close()

def create_word_in_col_head():
    f = open('col_header.txt', mode='r')
    fw = open('word_in_clo_header.txt', mode='w')
    dict_word = {}
    for line in f:
        r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = line.replace('\n', ' ')
        # line = line.replace('\n',' ').lower()
        line = re.sub(r1, ' ', line)
        for var in line.split(' '):
            if var not in dict_word:
                dict_word[var] = 1
                # fw.write(var+'\n')
            else:
                dict_word[var] += 1
    #print(dict_word)
    #print(len(dict_word))
    dict_important_word = {}
    for var in dict_word:
        if dict_word[var] == 1 or dict_word[var] == 1:
            fw.write(var + '\n')
            dict_important_word[var] = 1

    dict_sort = list(sorted(dict_word.items(), key=lambda d: d[1], reverse=False))

    #print(dict_sort)
    f = open('col_header.txt', mode='r')
    for line in f:
        r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = line.replace('\n', ' ')
        # line = line.replace('\n',' ').lower()
        line = re.sub(r1, ' ', line)
        s = ""
        for var in line.split():
            if var in dict_important_word:
                s += ' ' + var
        #print(s)
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
    f_dict_word = open('word_in_clo_header.txt', mode='r')
    dict_word = {}
    count = 0
    for var in f_dict_word:
        var = var.replace('\n', '').lower()
        dict_word[var] = count
        count += 1
    # print(dict_word)
    # print(len(dict_word))
    # print(sorted(dict_word))
    fr = open('data_word/' + name, mode='r')
    fw = open('wordkey_data/' + name, mode='w')

    for line in fr:
        line_var = ''

        for var in line.split(' '):
            var = var.replace('\n', '')
            if var in dict_word:
                line_var += ' ' + str(dict_word[var])
                if var not in dict_in_dataset:
                    dict_in_dataset[var] = 1
                else:
                    dict_in_dataset[var] += 1
        fw.write(line_var.strip() + '\n')
    #print(dict_in_dataset)
    #print(len(dict_in_dataset))

def logkey_count():
    fr_train = open('wordkey_data/hdfs_train', mode='r')
    fr_test_normal = open('wordkey_data/hdfs_test_normal', mode='r')
    fr_test_abnormal = open('wordkey_data/hdfs_test_abnormal', mode='r')

    dict_in_dataset = {}
    for line in fr_train:
        line_var = ''
        for var in line.split(' '):
            var = var.replace('\n', '')
            if var not in dict_in_dataset:
                dict_in_dataset[var] = 1
            else:
                dict_in_dataset[var] += 1
    print(len(dict_in_dataset))
    print(dict_in_dataset)
    for line in fr_test_normal:
        line_var = ''
        for var in line.split(' '):
            var = var.replace('\n', '')
            if var not in dict_in_dataset:
                dict_in_dataset[var] = 1
            else:
                dict_in_dataset[var] += 1
    print(len(dict_in_dataset))
    print(dict_in_dataset)

    for line in fr_test_abnormal:
        line_var = ''
        for var in line.split(' '):
            var = var.replace('\n', '')
            if var not in dict_in_dataset:
                dict_in_dataset[var] = 1
            else:
                dict_in_dataset[var] += 1
    print(len(dict_in_dataset))
    print(dict_in_dataset)

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

def log_to_dataframe(log_file, regex, headers):
    """ Function to transform log file to dataframe
    """
    log_messages = []
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in fin.readlines():
          #  print(line)
            try:
           #     print(line.split())
                match = regex.search(line.strip())
            #    print(match)
                message = [match.group(header) for header in headers]
               # print(message)
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass

    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    #print(logdf)
    return logdf

def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    #print(headers)
    #print(regex)
    return headers, regex

# f_word = open('data_noise/word_table',mode='r')
# list_word = []
# for line in f_word:
#     line = line.replace('\n','')
#     list_word.append(line)

def add_a_word_in_log(s):
    length_w = len(list_word)
    int_a =  random.randint(0, 100)
    if int_a % 2 == 0:
        s = s +' '+ list_word[random.randint(0,length_w-1)]
    else:
        s =  list_word[random.randint(0,length_w-1)] +' ' +s

    int_a =  random.randint(0, 100)
    if int_a % 2 == 0:
        s = s +' '+ 'to'
    else:
        s = 'to'+' ' +s
    return  s

# f_word_1 = open('data_noise/word_table',mode='r')
# dict_word = {}
# for line in f_word_1:
#     line = line.replace('\n','')
#     dict_word[line] = 1

def del_a_word_in_log(s):
    list_s = s.replace('\n','').split(' ')
    var = len(list_s)-1
    while var >= 0:
        if list_s[var] not in list_word and 'blk' not in list_s[var] :
            list_s.remove(list_s[var])
            break
        var= var -1
    s = ' '.join(list_s)
    return  s

def log_to_head_content(s, regex, headers):
    match = regex.search(s.strip())
    message = [match.group(header) for header in headers]
    #print(message)
    content = message[5]
    header = message[0] +" "+ message[1] +" "+  message[2] +" "+  message[3]+" "+ message[4]
   # print(header)
   # print(content)
    return header, content

def add_noise(sorce_file,p,save_file):
    source_f = open(sorce_file,mode='r')
    save_f = open(save_file,mode='w')
    p_add = p*3.0/4
    p_del = p*1.0/4

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

def group(noise_csv, lable_csv, map_file , normal_file,abnormal_file):
    f_normal = open(normal_file,mode='w')
    f_abnormal = open(abnormal_file,mode='w')
    map_f = open(map_file,mode='r')


    dict_map = {}
    for line in map_f:
        line = line.replace('\n','').split(' ')
        dict_map[line[1]] = line[0]

    dict_blk_id = {}
    dict_blk_id_label = {}

    csv_lable = pd.read_csv(lable_csv)
    csv_lable = csv_lable.values.tolist()
    for csv_lable_line in csv_lable:
        dict_blk_id[csv_lable_line[0]] = ''
        dict_blk_id_label[csv_lable_line[0]] = csv_lable_line[1]

    with open(noise_csv) as f:
        reader = csv.reader(f)
        count = 0
        for var in reader:
            count += 1
            if(count %100000 == 0):
                print(count)
            block_id,s = get_blk_id(' '+var[6]+'\n')
            if block_id in dict_blk_id:
                dict_blk_id[block_id] = dict_blk_id[block_id] + ' ' + var[7]#Spell drain
                #dict_blk_id[block_id] = dict_blk_id[block_id] + ' ' + var[8] #MoLFI
                #print(block_id)
            else:
                print(block_id)
                print(var[6])

    for var in dict_blk_id:
        line = str(dict_blk_id[var]).replace('\n','').split(' ')
        logkeySeq = ''
        for var2 in line:
            if var2 != '' :
                logkeySeq +=' '+ dict_map[var2]
        if dict_blk_id_label[var] == 'Normal':
            f_normal.write(logkeySeq+'\n')
        else:
            f_abnormal.write(logkeySeq+'\n')





def split_data_top(source_file,top ,save_file):
    f = open(source_file,mode='r')
    f_w = open(save_file,mode='w')
    total = 11175629
    count = 0
    for line in f:
        if(count < total*top):
            f_w.write(line)
        count+=1

def filte_log(source_file,save_file):
    source_f = open(source_file,mode='r')
    save_f = open(save_file,mode='w')

    for line in source_f:
        line = line.replace('\n','')
        r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = re.sub(r1, ' ', line)
        save_f.write(line+'\n')

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
   # print(model)
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

def csvtemplates_to_templates(csv_templates,templates):
    templates_f = open(templates,mode='w')
    with open(csv_templates) as f:
        reader = csv.reader(f)
        for var in reader:
            templates_f.write(var[0]+','+var[1]+'\n')

def replace_file(s):
    s = s.split(' ')
    str = ''
   # print(s)
    for var in s:
        if '/' not in var and '_' not in var :
            str +=' '+ var
    #print(str)
    return  str

def get_importain_word_Tf_idf(word2vec,name,glove_file,top_k,save_file,save_tmp_file):
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec, binary=False)
    glove_model = KeyedVectors.load_word2vec_format(glove_file, binary=False)
    f_s_tmp = open(save_tmp_file,mode='w')
    fr = open(name,mode='r')

    corpus = []
    for line in fr:
       # print(line)
        line = line.replace('\n',' ').replace(',',' ')
        line = line[8:] #sub head
        line = replace_file(line)
        r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = re.sub(r1, ' ' , line).lower()
        line2 = ''
        for var in line.split(' '):
               line2 += ' '+var
        corpus.append(line2)
        #print(line2)
    vectorizer = CountVectorizer(lowercase=False)
    X = vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    #print(tfidf.toarray())
    weight = tfidf.toarray()
    word_not_in_word2vec_mode = {}
   # top_k = 3
    fw = open(save_file,mode='w')
    list_save = []

    for i in range(len(weight)):
       # print(weight[i])
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

            list_x = []
            for var in x:
                list_x.append(str(var))
            list_x = " ".join(list_x)
            event_vector += list_x
            if count < top_k-1:
                #print(j)
                event_vector += ' '
            count += 1
        #fw.write(str(i+1) + ' ' + event_vector + '\n')
       # print(str(i+1)+' '+s+' '+s_w)
        f_s_tmp.write(str(i+1)+' '+s+' '+s_w+'\n')
        list_save.append(str(i+1) + ' ' + event_vector + '\n')

    fw.write(str(len(list_save))+' '+ str(top_k*100)+'\n')
    for line in list_save:
        fw.write(line)
    print(word_not_in_word2vec_mode)
    #print(len(word_not_in_word2vec_mode))

def get_map_logkey_to_num(template_file,num_file):
    f_tem = open(template_file,mode='r')
    num_file = open(num_file,mode='w')
    count = 1
    for line in f_tem:
        line = line.split(',')
        num_file.write(str(count)+' '+line[0]+'\n')
        count+=1

def map_logkey_to_num(map_file,source_file,save_file):
    source_f = open(source_file,mode='r')
    save_f = open(save_file,mode='w')
    map_f = open(map_file,mode='r')
    dict_map = {}

    for line in map_f:
        line = line.replace('\n','').split(' ')
        dict_map[line[1]] = line[0]

    for line in source_f:
        line = line.replace('\n','').split(' ')
        #print(line)
        s = ''
        for var in line:
            if var != '' :
                s +=' '+ dict_map[var]
        save_f.write(s+'\n')

def train_dataset_split(total_file,p,train_file,test_file):
    total_f = open(total_file,mode='r')
    train_f = open(train_file,mode='w')
    test_f = open(test_file,mode='w')

    count = 1

    list_train = []
    for line in total_f:
        list_train.append(line)
    
    total = len(list_train)
    list_train = np.array(list_train)
    # indexes = shuffle(np.arange(list_train.shape[0]))
    # list_train = list_train[indexes]

    for line in list_train:
        if count < total*p:
            train_f.write(line)
        else:
            test_f.write(line)
        count+=1

def simple_dataset(total_file,p,train_file):
    total_f = open(total_file,mode='r')
    train_f = open(train_file,mode='w')

    list_train = []
    for line in total_f:
        #line = line.replace('\n','')
        list_train.append(line)

    total = len(list_train)
    list_train = np.array(list_train)
    indexes = shuffle(np.arange(list_train.shape[0]))
    list_train = list_train[indexes]

    count = 1
    for line in list_train:
        if count < total*p:
            train_f.write(line)
        count+=1




def save_logkey_in_dataset_normal(file_source,file_save):
    dict_logkey_in_f = {}
    f = open(file_source,mode='r')
    f_save = open(file_save,mode='w')
    for line in f:
        line = line.replace('\n','').split(' ')
        for var in line:
            if var != '':
                logkey = int(var)
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
        line = line.split(' ')
        for var in line:
           if var != '':
                logkey = int(var)
                dict_logkey_in_f[logkey] = 1
    for line in f2:
        line = line.replace('\n','')
        line = line.split(' ')
        for var in line:
           if var != '':
                logkey = int(var)
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
    #print(map_dict)

    for line in source_f:
        line = line.replace('\n','').split(' ')
        s = ''
        for var in line:
            if var in map_dict:
                s = s+' '+ map_dict[var]
        save_f.write(s+'\n')

def unknown_logkey_to_0(train_logkey,file_sour,file_save):
    f_train = open(train_logkey,mode='r')
    f_s = open(file_sour,mode='r')
    f_save = open(file_save,mode='w')

    dict_train_logkey = {}
    for line in f_train:
        key = line.replace('\n','')
        dict_train_logkey[key] = 1

    #print(dict_train_logkey)
    for line in f_s:
        a = line.replace('\n','').split(' ')
        s = ''
        for var in a:
            if var in dict_train_logkey:
                s += ' ' + var
            else:
                if var != '':
                    s += ' ' +'0'
        #print(s)
        f_save.write(s+'\n')

def get_importain_word_txt(name, top_k,save_tmp_file):
    f_s_tmp = open(save_tmp_file, mode='w')
    fr = open(name, mode='r')
    list_temp = []
    corpus = []
    for line in fr:
        line = line.replace('\n', ' ').replace(',', ' ')
        #line = line[8:]  # sub head
        line = replace_file(line)
        r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = re.sub(r1, ' ', line).lower()
        line2 = ''
        line = line.replace('\\','')
        print(line.split(' '))
        list_temp.append(line.split(' '))
        for var in line.split(' '):
            var = var.replace('\\','')
            line2 += ' ' + var
        print(line2)
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

def get_importain_word_csv(name, top_k,save_tmp_file):
    f_s_tmp = open(save_tmp_file, mode='w')
    fr = open(name, mode='r')
    list_temp = []
    corpus = []
    for line in fr:
        #print(line)
        line = line.replace('\n', ' ').replace(',', ' ')
        line = line[8:]  # sub head
        line = replace_file(line)
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

def similartiy_template(unknow,exist):
    logkey_unknow = unknow.split(' ')[0]
    logkey_exist = exist.split(' ')[0]
    r1 = '[0-9’!"#$%&\'\\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    unknow = re.sub(r1, '', unknow).lower().strip()
    exist = re.sub(r1, '', exist).lower().strip()

    #print(unknow)
    #print(exist)

    list_unknow = unknow.split(' ')
    list_exist = exist.split(' ')

    count = 0
    length = 0
    #print(list_unknow)
    #print(list_exist)
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


    #if count*2.0/length >= 1:
        #print(count)
        #print(length)
        #print(list_unknow)
        #print(list_exist)

    #print("length=%d count=%d value=%f"%(length,count,count*1.0/length))
    return logkey_unknow,logkey_exist,count*2.0/length

def similartiy_template_0(unknow,exist):
    logkey_unknow = unknow.split(' ')[0]
    logkey_exist = exist.split(' ')[0]
    r1 = '[0-9’!"#$%&\'\\()+,-./:;=?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    unknow = re.sub(r1, '', unknow).lower().strip()
    exist = re.sub(r1, '', exist).lower().strip()

    #print(unknow)
    #print(exist)

    list_unknow = unknow.split(' ')
    list_exist = exist.split(' ')

    count = 0
    length = 0
    #print(list_unknow)
    #print(list_exist)
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


    #if count*2.0/length >= 1:
        #print(count)
        #print(length)
        #print(list_unknow)
        #print(list_exist)

    #print("length=%d count=%d value=%f"%(length,count,count*1.0/length))
    return logkey_unknow,logkey_exist,count*2.0/length

def get_map_unkonw_tmp_to_exist(exist_t,unknow_t,save_map):
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
            logkey_unknow,logkey_exist, value = similartiy_template(line,exist)
            dict_similarity[logkey_exist] = value
        #print(dict_similarity)
        dict_similarity = sorted(dict_similarity.items(), key=lambda d: d[1] ,reverse=True )
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
        print(dict_similarity[0][1])
        if dict_similarity[0][1] >= th:
            f_map.write(line.split(' ')[0]+' ' +str(dict_similarity[0][0])+'\n')
        else:
            f_map.write(line.split(' ')[0] + ' 0\n')

def count_map_to_exist_logkey(exist_f):
    f = open(exist_f,mode='r')
    dict_word_logkey = {}
    for line in f:
        line = line.replace('\n','').split(' ')
        dict_word_logkey[line[1]] = 1
    dict_word_logkey = sorted(dict_word_logkey)
    print(len(dict_word_logkey))
    print(dict_word_logkey)

def group_cmp(noise_csv,lable_csv, save_normal_file):
    f_normal = open(save_normal_file,mode='w')

    csv_lable = pd.read_csv(lable_csv)
    print(csv_lable)
    csv_lable = csv_lable.values.tolist()
    list_blkid = []

    dict_blk_id = {}
    for csv_lable_line in csv_lable:
        dict_blk_id[csv_lable_line[0]] = ''
        list_blkid.append(csv_lable_line[0])


    with open(noise_csv) as f:
        reader = csv.reader(f)
        count = 0
        for var in reader:
            count += 1
            if(count %100000 == 0):
                print(count)
            block_id,s = get_blk_id(' '+var[6]+'\n')
            if block_id in dict_blk_id:
                dict_blk_id[block_id] = dict_blk_id[block_id] + ' ' + var[7]
                #print(block_id)

    for i in range(len(list_blkid)):
        var = list_blkid[i]
        f_normal.write( str(dict_blk_id[var])+'\n')

# f = open('data_noise/col_header.txt',mode='r')
# dict_logkey={}
# for line in f:
#     line = line.rstrip()
#    # print(line)
#     obj = re.match(r'([0-9]+).(.*)',line)
#     dict_logkey[obj.group(1)] = obj.group(2)
# #print(dict_logkey)

def match_logkey(s):

    for var in dict_logkey:
        obj = re.search(dict_logkey[var],s)
        if obj!= None:
            return var
    #print(s)
    return '0'

def group_row(noise_csv,lable_csv, save_normal_file , save_abnormal_file):
    f_normal = open(save_normal_file,mode='w')
    f_abnormal = open(save_abnormal_file,mode='w')

    csv_lable = pd.read_csv(lable_csv)
   # print(csv_lable)
    csv_lable = csv_lable.values.tolist()

    dict_blk_id = {}
    list_blkid = []
    dict_blk_id_label = {}

    for csv_lable_line in csv_lable:
        dict_blk_id[csv_lable_line[0]] = ''
        list_blkid.append(csv_lable_line[0])
        dict_blk_id_label[csv_lable_line[0]] = csv_lable_line[1]

    f = open(noise_csv, mode='r') #将数据按照blk_id进行group
    j= 0
    for line in f.readlines():
        if j %100000 == 0:
            print(j)
        blk_id,s = get_blk_id(line)
        line = sub_head(line)
        #print(blk_id)
        if blk_id in dict_blk_id:
            dict_blk_id[blk_id] = dict_blk_id[blk_id] + line
        j = j+1

    for i in range(len(list_blkid)):
        if i %1000 == 0:
            print(i)
        line_blk_vac = ''
        key = list_blkid[i]
        for line in dict_blk_id[key].split('\n'):
            line = line.strip()
            if line != '\n' and line != '':
                line_blk_vac += ' '+ match_logkey(line)
        if dict_blk_id_label[key] == 'Normal':
            f_normal.write(line_blk_vac+'\n')
        else:
            f_abnormal.write(line_blk_vac + '\n')

def count_deeplog_0(normal_f,abnormal_f):
    f_n = open(normal_f,mode='r')
    f_ab = open(abnormal_f,mode='r')

    count_logkey = {}
    total = 0
    total_0 = 0
    for line in f_n:
        line = line.replace('\n','').split(' ')
        total += 1
        for var in line:
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

    print("total=%d  total_0=%d %f"%(total,total_0,total_0*100.0/total))

def compare_two_file(row_file,noise_file):
    f_row = open(row_file,mode='r')
    noise_f = open(noise_file,mode='r')

    list_row = []
    list_noise=[]
    for line in f_row:
        list_row.append(line.replace('\n',''))
    for line in noise_f:
        list_noise.append(line.replace('\n',''))

    for i in range(len(list_row)):
        row_line = list_row[i].split(' ')
        noise_line = list_noise[i].split(' ')
        for j in range(len(row_line)):
            if row_line[j] != noise_line[j]  and row_line[j] != '11':
                print(row_line)
                print(noise_line)

def count_logky(normal_f):
    f_n = open(normal_f,mode='r')
    count_logkey = {}
    total = 0
    for line in f_n:
        line = line.replace('\n','').split(' ')
        for var in line:
            if var != '':
                total += 1
                if var not in count_logkey:
                    count_logkey[var] = 1
                else:
                    count_logkey[var] += 1
    for var in count_logkey:
        print("var=%s  %f"%(var,count_logkey[var]*100.0/total))

def count_total(train_f,normal_f,abnormal_f):
    f_n = open(normal_f,mode='r')
    f_ab = open(abnormal_f,mode='r')
    f_tr = open(train_f,mode='r')

    count_logkey = {}
    total = 0
    for line in f_n:
        line = line.replace('\n','').split(' ')
        for var in line:
            if var != '':
                total += 1
                if var not in count_logkey:
                    count_logkey[var] = 1
                else:
                    count_logkey[var] += 1
    for line in f_tr:
        line = line.replace('\n','').split(' ')
        for var in line:
            if var != '':
                total += 1
                if var not in count_logkey:
                    count_logkey[var] = 1
                else:
                    count_logkey[var] += 1

    for line in f_ab:
        line = line.replace('\n','').split(' ')
        for var in line:
            if var != '':
                total += 1
                if var not in count_logkey:
                    count_logkey[var] = 1
                else:
                    count_logkey[var] += 1

    for var in count_logkey:
        if var == '0':
            print("var=%s %d %d %f"%(var,count_logkey[var],total,count_logkey[var]*100.0/total))

def get_map_0_to_exist(p_num,map_0,map_exist,map_0_new):
    f_map_0 = open(map_0,mode='r')
    f_map_exist = open(map_exist,mode='r')
    f_map_0_new = open(map_0_new,mode='w')
    dict_map_0 = {}
    dict_map_exist = {}
    total_0 = 0
    total = 0
    list_map_0 = []
    for line in f_map_0:
        line = line.replace('\n','').split(' ')
        dict_map_0[line[0]] = line[1]
        list_map_0.append(line)
        if line[1] == '0':
            total_0 += 1
        total += 1

    for line in f_map_exist:
        line = line.replace('\n', '').split(' ')
        dict_map_exist[line[0]] = line[1]


    i =   0
    dict_map_0_new = {}
    for var in dict_map_0:
        if dict_map_0[var] == '0' and i < p_num*total_0:
            dict_map_0_new[var] = dict_map_exist[var]
            i += 1
        else:
            dict_map_0_new[var] = dict_map_0[var]
    for var in dict_map_0_new:
        f_map_0_new.write(var+' '+dict_map_0_new[var]+'\n')

    '''
    i = 0
    for i in range(int(p_num*total)):
        index =  random.randint(0,total - 1)
        logkey = list_map_0[index][0]
        dict_map_0[logkey] = dict_map_exist[logkey]
    
    print("total=%d total_0=%d %f new=%d"%(total,total_0,total_0*1.0/total , int(total*p_num)))
    for var in dict_map_0:
        f_map_0_new.write(var+' '+dict_map_0[var]+'\n')
    '''

def count_logkey_0_in_map_0(map_0):
    f = open(map_0,mode='r')
    total = 0
    total_0 = 0
    for line in f:
        line = line.replace('\n','').split(' ')
        total += 1
        if line[1] == '0':
            total_0+=1
    print('total=%d total_0=%d %f'%(total,total_0,total_0*1.0/total))


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


if __name__ == "__main__":
    input_dir = sys.argv[1]

    csvtemplates_to_templates(input_dir + '/HDFS.log_templates.csv', input_dir + '/templates')
    get_map_logkey_to_num(input_dir + '/templates', input_dir + '/map_logkey_to_num')

    group(input_dir + '/HDFS.log_structured.csv', 'anomaly_label.csv', input_dir + '/map_logkey_to_num',  input_dir + '/normal',input_dir + '/abnormal')
    train_dataset_split( input_dir + '/normal', 0.7 , input_dir + '/train_normal',input_dir + '/test_normal')


    simple_dataset( input_dir + '/test_normal', 0.01 , input_dir + '/simple_test_normal')
    simple_dataset( input_dir + '/abnormal', 0.01 , input_dir + '/simple_abnormal')
