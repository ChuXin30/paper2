#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' This is a demo file for the Invariants Mining model.
    API usage:
        dataloader.load_HDFS(): load HDFS dataset
        feature_extractor.fit_transform(): fit and transform features
        feature_extractor.transform(): feature transform after fitting
        model.fit(): fit the model
        model.predict(): predict anomalies on given data
        model.evaluate(): evaluate model accuracy with labeled data
'''

import sys
sys.path.append('../')
from loglizer.models import InvariantsMiner
from loglizer import dataloader, preprocessing
import numpy
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

struct_log_test = '../data/HDFS/HDFS.log_structured.csv' # The structured log file
struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file
epsilon = 0.5 # threshold for estimating invariant space
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

    (x_train, _), (x_test, _) = dataloader.load_HDFS(struct_log,
                                                     window='session', 
                                                     train_ratio=1,
                                                     split_type='sequential')
    print(type(x_train))
    # Feature extraction
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train)

    # Model initialization and training
    model = InvariantsMiner(epsilon=epsilon)
    model.fit(x_train)

    # Predict anomalies on the training set offline, and manually check for correctness
    y_train = model.predict(x_train)

    # Predict anomalies on the test set to simulate the online mode
    # x_test may be loaded from another log file
    x_test = feature_extractor.transform(x_test)
    y_test = model.predict(x_test)

    # If you have labeled data, you can evaluate the accuracy of the model as well.
    # Load structured log with label info
    #(x_train, x_train), (x_test, y_test) = dataloader.load_HDFS(struct_log_test,label_file=label_file,window='session',train_ratio=0.00,split_type='sequential')
    #print(x_test[0])
    #print(y_test[0])
    #print(type(x_test[0]))
    #print(type(x_test))
    #print(len(x_test))
    #np.save("x_train.npy", x_train)
    #np.save("y_train.npy", x_train)
    #np.save("x_test.npy", x_test)
    #np.save("y_test.npy", y_test)

    x_test = np.load("x_test.npy",allow_pickle=True)
    y_test = np.load("y_test.npy")
    #print(x_test[0])
    #print(y_test[0])
    #print(type(x_test[0]))
    #print(type(x_test))
    #print(len(x_test))


    #x_test = feature_extractor.transform(x_test)
    #precision, recall, f1 = model.evaluate(x_test, y_test)

    x_test = feature_extractor.transform(x_test)
    x_test = model.predict(x_test)
    print(classification_report( y_test.tolist(),x_test.tolist()))
    print(confusion_matrix( y_test.tolist(),x_test.tolist()))
    '''
    #(x_train, _), (x_test, _) = dataloader.load_HDFS(struct_log,window='session',train_ratio=0.5,split_type='sequential')
    x_train,y_train = generate('hdfs_train')

    # Feature extraction
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train)

    # Model initialization and training
    model = InvariantsMiner(epsilon=epsilon)
    model.fit(x_train)

    x_test, y_test = generate('hdfs_test')
    x_test = feature_extractor.fit_transform(x_test)
    x_test = model.predict(x_test)

    print(classification_report(x_test.tolist(), y_test.tolist()))
    print(confusion_matrix(x_test.tolist(), y_test.tolist()))
    '''