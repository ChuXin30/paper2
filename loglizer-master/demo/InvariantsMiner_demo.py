#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import InvariantsMiner
from loglizer import dataloader, preprocessing
import numpy

#struct_log = '../data/HDFS/HDFS.log_structured.csv' # The structured log file

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
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='sequential')

    #x_test,y_test =  generate('hdfs_test')
    #x_train,y_train = generate('hdfs_train')
    print(len(x_test))
    print(len(y_test))
    print(x_test[0])
    print(y_test[0])

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train)
    x_test = feature_extractor.transform(x_test)

    model = InvariantsMiner(epsilon=epsilon)
    model.fit(x_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)
    
    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)

