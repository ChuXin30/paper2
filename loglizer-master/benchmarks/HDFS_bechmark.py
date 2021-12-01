#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
import pandas as pd
from loglizer.models import *
from loglizer import dataloader, preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy

#run_models = ['PCA', 'InvariantsMiner', 'LogClustering', 'IsolationForest', 'LR', 'SVM', 'DecisionTree']
#run_models = ['PCA', 'LogClustering','InvariantsMiner']
#run_models = ['PCA','LogClustering','InvariantsMiner']
run_models = ['PCA','LogClustering']

#run_models = ['InvariantsMiner']

#struct_log = '../data/HDFS/HDFS.npz' # The benchmark dataset
#struct_log = '../data/HDFS/HDFS.log_structured.csv' # The structured log file
#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
#label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file
def generate(normal_train,normal_test,abnormal_test):
    normal_trian_f = open(normal_train,mode='r')
    normal_test_f = open(normal_test,mode='r')
    abnormal_test_f = open(abnormal_test,mode='r')

    list_train = []
    list_train_lable = []
    list_test = []
    list_test_lable = []
    for line in normal_trian_f:
        list_train.append(line.strip().replace('\n','').split(' '))
        list_train_lable.append(0)

    for line in normal_test_f:
        list_test.append(line.strip().replace('\n','').split(' '))
        list_test_lable.append(0)

    for line in abnormal_test_f:
        list_test.append(line.strip().replace('\n','').split(' '))
        list_test_lable.append(1)

    return  numpy.array(list_train),numpy.array(list_train_lable),numpy.array(list_test),numpy.array(list_test_lable)


#x_tr, y_train, x_te, y_test = generate('/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_0/WITH_S_train',
#                                       '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_0/Deeplog_test_normal',
#                                       '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_0/Deeplog_test_abnormal')



#hdfs_train = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_0/WITH_S_train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_0/Deeplog_test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_0/Deeplog_test_abnormal'

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/row_dataset/hdfs_train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/row_dataset/hdfs_test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/row_dataset/hdfs_abnormal'

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/MoLFI_noise_0/WITH_S_train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/MoLFI_noise_0/Deeplog_test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/MoLFI_noise_0/Deeplog_test_abnormal'

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data/hdfs_train' #0.01633 0.10837
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data/hdfs_test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data/hdfs_test_abnormal'

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_30/WITH_S_train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_30/Deeplog_test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_30/Deeplog_test_abnormal'

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_30/WITH_S_train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_0/WITH_S_test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_0/WITH_S_test_abnormal'

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_30/WITH_S_train' #0.42400  0.39288
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_50/Deeplog_test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_50/Deeplog_test_abnormal'

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_30/WITH_S_train' #0.47411 0.76559
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_50/WITH_S_test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/drain_noise_50/WITH_S_test_abnormal'

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_5/WITH_S_train' #0.51865 0.76736
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_0.016/Deeplog_test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_0.016/Deeplog_test_abnormal'

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_5/WITH_S_train' #0.51872 0.76772
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_line_5/Deeplog_test_normal'#
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_line_5/Deeplog_test_abnormal'

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_5/WITH_S_train' #0.28930 0.62536
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_line_10/Deeplog_test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_line_10/Deeplog_test_abnormal'

'''
#hdfs
hdfs_train = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_5/WITH_S_train' #
hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_line_15/Deeplog_test_normal'
hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_line_15/Deeplog_test_abnormal'

x_tr, y_train, x_te, y_test = generate(hdfs_train,
                                       hdfs_test_normal,
                                       hdfs_test_abnormal)
print(hdfs_test_normal)
'''
def generate_bgl(window_size,normal_train, normal_test, abnormal_test):
    normal_trian_f = open(normal_train, mode='r')
    normal_test_f = open(normal_test, mode='r')
    abnormal_test_f = open(abnormal_test, mode='r')

    list_train = []
    list_train_lable = []
    list_test = []
    list_test_lable = []

    count = 0
    a_window_list = []
    for line in normal_trian_f:
        a_window_list.append(line.replace('\n',''))
        count += 1
        if count%(window_size+1)  == 0:
            list_train.append(a_window_list)
            list_train_lable.append(0)
            a_window_list = []

    print("train length=%d"%(len(list_train)))
    count = 0
    a_window_list = []
    for line in normal_test_f:
        a_window_list.append(line.replace('\n',''))
        count += 1
        if count%(window_size+1)  == 0:
            list_test.append(a_window_list)
            list_test_lable.append(0)
            a_window_list = []

    print("test normal length=%d"%(len(list_test)))

    count = 0
    a_window_list = []
    for line in abnormal_test_f:
        a_window_list.append(line.replace('\n',''))
        count += 1
        if count%(window_size+1)  == 0:
           # print(a_window_list)
            list_test.append(a_window_list)
            list_test_lable.append(1)
            a_window_list = []

    print("test total length=%d"%(len(list_test)))

    return numpy.array(list_train), numpy.array(list_train_lable), numpy.array(list_test), numpy.array(
        list_test_lable)

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_1/train' #0.00218 0.23662
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_1/test_new'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_1/abnormal_new'
#window_size = 5

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_80_window_20/train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_80_window_20/test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_80_window_20/test_abnormal'
#window_size = 10

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_80_window_20/train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_80_window_20/test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_80_window_20/test_abnormal'
#window_size = 15

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_80_window_20/train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_80_window_20/test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_80_window_20/test_abnormal'
#window_size = 20


#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/random_dataset_20/train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/random_dataset_20/test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/random_dataset_20/test_abnormal'
#window_size = 20

#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/order_random_dataset_20/train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/order_random_dataset_20/test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/order_random_dataset_20/test_abnormal'
#window_size = 20

#order_random_dataset_20
#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/order_random_dataset_20/train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/order_random_dataset_20/test_normal_new'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/order_random_dataset_20/test_abnormal_new'
#window_size = 20

#dataset_80_exist_0
#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_80_exist_0/train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_80_exist_0/test_normal_new'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/dataset_80_exist_0/test_abnormal_new'
#window_size = 20

#order_random_dataset_20_exist_0
#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/order_random_dataset_20_exist_0/train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/order_random_dataset_20_exist_0/test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/order_random_dataset_20_exist_0/test_abnormal'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/order_random_dataset_20_exist_0/test_normal_new'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/order_random_dataset_20_exist_0/test_abnormal_new'
#window_size = 20

#drain_dataset_80
#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_dataset_80/train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_dataset_80/test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_dataset_80/test_abnormal'
#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_dataset_80/train_simple'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_dataset_80/test_normal_simple'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_dataset_80/test_abnormal_simple'
#window_size = 20

#drain_with_level_80
#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_with_level_80/train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_with_level_80/test_normal'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_with_level_80/test_abnormal'
#window_size = 20

#drain_random_with_level_20 noise_0
#hdfs_train = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_random_with_level_20/train'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_random_with_level_20/test_normal'#0.00032 0.98348
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_random_with_level_20/test_abnormal'
#hdfs_test_normal = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_random_with_level_20/test_normal_0'
#hdfs_test_abnormal = '/home/hao/Desktop/论文２　实验代码/BGL/data/drain_random_with_level_20/test_abnormal_0'
window_size = 20
#noise_0 BGL
#hdfs_train = '/media/hao/file/paper2_code/BGL/noise/drain_with_level_80/train'
#hdfs_test_normal = '/media/hao/file/paper2_code/BGL/noise/drain_with_level_80/test_normal'
#hdfs_test_abnormal = '/media/hao/file/paper2_code/BGL/noise/drain_with_level_80/test_abnormal'

#bgl noise
#hdfs_train = '/media/hao/file/paper2_code/BGL/noise/drain_random_with_level_20/train'
#hdfs_test_normal = '/media/hao/file/paper2_code/BGL/noise/drain_random_with_level_20/test_normal_simple'
#hdfs_test_abnormal = '/media/hao/file/paper2_code/BGL/noise/drain_random_with_level_20/test_abnormal_simple'
#x_tr, y_train, x_te, y_test = generate_bgl(window_size,hdfs_train,hdfs_test_normal,hdfs_test_abnormal)


if __name__ == '__main__':
    print(sys.argv)
    hdfs_train = sys.argv[1]
    hdfs_test_normal = sys.argv[2]
    hdfs_test_abnormal = sys.argv[3]
    print(hdfs_train)
    print(hdfs_test_normal)
    print(hdfs_test_abnormal)
    x_tr, y_train, x_te, y_test = generate_bgl(window_size, hdfs_train, hdfs_test_normal, hdfs_test_abnormal)

    #(x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(struct_log,label_file=label_file,window='session',train_ratio=0.5,split_type='uniform')
    #(x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(struct_log,window='session',train_ratio=0.5,split_type='uniform')
    #(x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(struct_log,window='session',train_ratio=0.1,split_type='uniform')


    '''
    (x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(struct_log,label_file=label_file,window='session',train_ratio=0.1,split_type='uniform')
    np.save("x_train_01.npy", x_tr)
    np.save("y_train_01.npy", y_train)
    np.save("x_test_01.npy", x_te)
    np.save("y_test_01.npy", y_test)
    x_te = np.load("x_test_01.npy",allow_pickle=True)
    y_test = np.load("y_test_01.npy",allow_pickle=True)
    x_tr = np.load("x_train_01.npy",allow_pickle=True)
    y_train = np.load("y_train_01.npy",allow_pickle=True)
    '''

#    (x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(struct_log,label_file=label_file,window='session',train_ratio=0.5,split_type='uniform')
    '''
    x_te = np.load("x_test.npy",allow_pickle=True)
    y_test = np.load("y_test.npy",allow_pickle=True)
    x_tr = np.load("x_train.npy",allow_pickle=True)
    y_train = np.load("y_train.npy",allow_pickle=True)
    '''


    #x_tr, y_train, x_te, y_test = generate('/home/hao/Desktop/论文２　实验代码/DeepLog-master/data/hdfs_train','/home/hao/Desktop/论文２　实验代码/DeepLog-master/data/hdfs_test_normal','/home/hao/Desktop/论文２　实验代码/DeepLog-master/data/hdfs_test_abnormal')
    #x_tr, y_train, x_te, y_test = generate('/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_20/WITH_S_train','/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_20/WITH_S_test_normal','/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_20/WITH_S_test_abnormal')
    #x_tr, y_train, x_te, y_test = generate('/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_20/WITH_S_train','/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_20/Deeplog_test_normal','/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_20/Deeplog_test_abnormal')

    '''
    normal_dataset = 4855+138
    x_tr = x_toral[0:normal_dataset]
    y_train = y_total[0:normal_dataset]
    x_te = x_toral[normal_dataset:]
    y_test = y_total[normal_dataset:]
    '''

    print(x_te[0]) #['09a53393', '3d91fa85', '09a53393', '09a53393', 'd38aa58d', 'e3df2680', 'e3df2680', '5d5de21c', 'd38aa58d', 'e3df2680', 'd38aa58d', '5d5de21c', '09a53393', '40651754', '728076ac', '5d5de21c', 'd6b7b743', '73c2ec69', '5d5de21c', 'dba996ef', 'd63ef163', 'd63ef163', 'd63ef163', 'dba996ef', 'dba996ef', 'dba996ef']
    print(y_test[0])#1
    print(x_tr[0])#['3d91fa85', '09a53393', '09a53393', '09a53393', '5d5de21c', 'd38aa58d', 'e3df2680', 'd38aa58d', 'e3df2680', '5d5de21c', '5d5de21c', 'd38aa58d', 'e3df2680', '626085d5', '81cee340', '626085d5', '626085d5', '81cee340', '626085d5', '626085d5', '81cee340', '626085d5', '626085d5', '81cee340', '626085d5', 'd63ef163', 'd63ef163', 'd63ef163', 'dba996ef', 'dba996ef', 'dba996ef']
    print(y_train[0])#0

    #print(len(x_tr))
    #print(len(x_tr[0:5000]))

    '''
    x_te = np.concatenate((x_te,x_tr[normal_dataset:]))
    y_test = np.concatenate((y_test,y_train[normal_dataset:]))

    x_tr = x_tr[0:normal_dataset]
    y_train = y_train[0:normal_dataset]
    '''
    #print(type(y_test))
   # print(y_test)
   # print(y_test[0])
    #print((len(y_test)))
    #print(len(x_te))
    #print(len(x_tr))

    benchmark_results = []
    for _model in run_models:
        print('Evaluating {} on HDFS:'.format(_model))
        if _model == 'PCA':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf', 
                                                      normalization='zero-mean')
            model = PCA()
            model.fit(x_train)
        
        elif _model == 'InvariantsMiner':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            #model = InvariantsMiner(epsilon=0.6) #0.23341
            #model = InvariantsMiner(epsilon=0.5) #0.00744
            #model = InvariantsMiner(epsilon=0.7) #
            #model = InvariantsMiner(epsilon=0.4) #
            model = InvariantsMiner(epsilon=0.3) #

            model.fit(x_train)

        elif _model == 'LogClustering':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
#            model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)#0.98
            #model = LogClustering(max_dist=0.4, anomaly_threshold=0.4)#0.99
            #model = LogClustering(max_dist=0.2, anomaly_threshold=0.2)
            #model = LogClustering(max_dist=0.1, anomaly_threshold=0.1)
            model = LogClustering(max_dist=0.05, anomaly_threshold=0.05)
            #model = LogClustering(max_dist=0.03, anomaly_threshold=0.03)

            model.fit(x_train[y_train == 0, :]) # Use only normal samples for training

        elif _model == 'IsolationForest':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = IsolationForest(random_state=2019, max_samples=0.9999, contamination=0.03, 
                                    n_jobs=4)
            model.fit(x_train)

        elif _model == 'LR':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = LR()
            model.fit(x_train, y_train)

        elif _model == 'SVM':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = SVM()
            model.fit(x_train, y_train)

        elif _model == 'DecisionTree':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = DecisionTree()
            model.fit(x_train, y_train)
        
        x_test = feature_extractor.transform(x_te)
        x_test = model.predict(x_test)
       # print(len(x_test))
       # print(len(y_test))
       # print(x_test[0])
       # print(y_test[0])

        print(classification_report(y_test.tolist(), x_test.tolist(),digits=5))
        print(confusion_matrix(y_test.tolist(), x_test.tolist()))
        #print('Train accuracy:')
        #precision, recall, f1 = model.evaluate(x_train, y_train)
        #benchmark_results.append([_model + '-train', precision, recall, f1])
        #print('Test accuracy:')
        #precision, recall, f1 = model.evaluate(x_test, y_test)
        #benchmark_results.append([_model + '-test', precision, recall, f1])

    pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1']) \
      .to_csv('benchmark_result.csv', index=False)
