#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' This is a demo file for the PCA model.
    API usage:
        dataloader.load_HDFS(): load HDFS dataset
        feature_extractor.fit_transform(): fit and transform features
        feature_extractor.transform(): feature transform after fitting
        model.fit(): fit the model
        model.predict(): predict anomalies on given data
        model.evaluate(): evaluate model accuracy with labeled data
'''
from sklearn.cross_validation import train_test_split
import sys
sys.path.append('../')
from loglizer.models import PCA
from loglizer import dataloader, preprocessing
import numpy as np
import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
#struct_log = '../data/HDFS/HDFS.log_structured.csv' # The structured log file
struct_log_test = '../data/HDFS/HDFS.log_structured.csv'
def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    dataset = []
    with open('../data/HDFS/' + name+'_abnormal', 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = list(map(int, line.strip().split()))
            inputs.append(line)
            outputs.append(1)
    print('Number of sessions({}): {}'.format(name+'_abnormal', num_sessions))

    with open('../data/HDFS/' + name+'_normal', 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = list(map(int, line.strip().split()))
            inputs.append(line)
            outputs.append(0)
    print('Number of sessions({}): {}'.format(name+'_normal', num_sessions))
    inputs = numpy.array(inputs)
    outputs = numpy.array(outputs)

    return (inputs,outputs)

if __name__ == '__main__':
    ## 1. Load strutured log file and extract feature vectors
    # Save the raw event sequence file by setting save_csv=True
    '''
    (x_train, _), (_, _) = dataloader.load_HDFS(struct_log, window='session',split_type='sequential', save_csv=True)
    print(len(x_train))
    print(x_train[0])
    print(len(x_train[0]))
    print(x_train)

    (x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(struct_log, window='session', train_ratio=0.5,  split_type='uniform')
    print(len(x_te))
    print(x_te[0])
    print(len(x_te[0]))
    print(len(y_test))
    print(y_test[0])
    '''
    #(x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
    #                                                            label_file=label_file,
    #                                                            window='session',
    #                                                            train_ratio=0.5,
    #                                                           split_type='uniform')
    x_train,y_train = generate('hdfs_train')
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf',normalization='zero-mean')
    
    ## 2. Train an unsupervised model
    print('Train phase:')
    # Initialize PCA, or other unsupervised models, LogClustering, InvariantsMiner
    model = PCA() 
    # Model hyper-parameters may be sensitive to log data, here we use the default for demo
    model.fit(x_train)
    # Make predictions and manually check for correctness. Details may need to go into the raw logs
    y_train = model.predict(x_train) 

    ## 3. Use the trained model for online anomaly detection
    print('Test phase:')
    # Load another new log file. Here we use struct_log for demo only
    (x_test, _), (_, _) = dataloader.load_HDFS(struct_log, window='session', split_type='sequential')
    # Go through the same feature extraction process with training, using transform() instead
    x_test = feature_extractor.transform(x_test)
    y_test = model.predict(x_test)
    # Finally make predictions and alter on anomaly cases
    print(len(x_test))
    print(x_test[0])

    x_test,y_test =  generate('hdfs_test')
    #print(len(x_test))
    #print(x_test[0])
    x_test = feature_extractor.transform(x_test)
    x_test = model.predict(x_test)

    print(x_test)
    print(y_test)
    print((len(x_test)))
    print((len(y_test)))
    print(x_test[500000])
    print(y_test[500000])
    print(x_test[500])
    print(y_test[500])
    print(type(x_test))
    print(type(y_test))

    #x_test=[int(x) for x in x_test]
    #x_test=numpy.array(x_test)
    print(x_test[0])
    print(x_test.shape)
    print(y_test.shape)
    #x_test = x_test.reshape(1,-1)
    #print(x_test.shape)
    #y_test = y_test.reshape(1,-1)
    #print(y_test.shape)

    #print('Train accuracy:')

    print(classification_report( y_test.tolist(),x_test.tolist()))
    print(confusion_matrix( y_test.tolist(),x_test.tolist()))
    #precision, recall, f1 = model.evaluate(x_test, y_test)
    #print("p={} r={} f1={}".format(precision,recall,f1))


