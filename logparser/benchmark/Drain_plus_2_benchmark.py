#!/usr/bin/env python

import sys

sys.path.append('../')
from logparser import Drain_plus_2, evaluator
import os
import pandas as pd
from layers.file_output_layer import FileOutputLayer
from layers.knowledge_layer import KnowledgeGroupLayer
from layers.mask_layer import MaskLayer
from layers.tokenize_group_layer import TokenizeGroupLayer
from layers.dict_group_layer import DictGroupLayer

import sys
from evaluator import evaluator
import os
import re
import string
import hashlib
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import argparse


input_dir = '../logs/'  # The input directory of log file
output_dir = 'Drain_plus_2_result/'  # The output directory of parsing results

def load_logs(log_file, regex, headers):
    """ Function to transform log file to dataframe
    """
    log_messages = dict()
    linecount = 0
    with open(log_file, 'r') as fin: 
        
        for line in tqdm(fin.readlines(), desc='load data'):
            try:
                linecount += 1
                match = regex.search(line.strip())
                message = dict()
                for header in headers:
                    message[header] = match.group(header)
                message['LineId'] = linecount
                log_messages[linecount] = message
            except Exception as e:
                pass
    return log_messages

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
    return headers, regex

benchmark_settings = {

    'HDFS_all': {
        'log_file': 'HDFS/HDFS.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'depth': 4,
        'tau': 0.7
    },

    'BGL_all': {
        'log_file': 'BGL/BGL.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
        'st': 0.5,
        'depth': 4,
        'tau': 1.1
    },


    # 'Spark100k': {
    #     'log_file': 'Spark/Spark_100k_match.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.5,
    #     'depth': 4,
    #     # 'tau': 0.55
    #     'tau': 0.7
    # },

    # 'Spark10k': {
    #     'log_file': 'Spark/Spark_10k_match.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.5,
    #     'depth': 4,
    #     # 'tau': 0.55
    #     'tau': 0.7
    # },

    # 'Spark1k': {
    #     'log_file': 'Spark/Spark_1k_match.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.5,
    #     'depth': 4,
    #     # 'tau': 0.55
    #     'tau': 0.7
    # },

    # 'Spark1m': {
    #     'log_file': 'Spark/Spark_1m_match.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.5,
    #     'depth': 4,
    #     # 'tau': 0.55
    #     'tau': 0.7
    # },



    # 'BGL1m': {
    #     'log_file': 'BGL/BGL_1m.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4,
    #     # 'tau': 0.75
    #     'tau': 1.1
    # },

    # 'BGL1k': {
    #     'log_file': 'BGL/BGL_1k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4,
    #     # 'tau': 0.75
    #     'tau': 1.1
    # },

    # 'BGL10k': {
    #     'log_file': 'BGL/BGL_10k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4,
    #     # 'tau': 0.75
    #     'tau': 1.1
    # },

    # 'BGL100k': {
    #     'log_file': 'BGL/BGL_100k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4,
    #     # 'tau': 0.75
    #     'tau': 1.1
    # },


    # 'HDFS1k': {
    #     'log_file': 'HDFS/HDFS_1k.log',
    #     # 'log_file': 'HDFS/HDFS_1k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.2,
    #     'depth': 4,
    #     'tau': 0.7
    # },

    # 'HDFS10k': {
    #     'log_file': 'HDFS/HDFS_10k.log',
    #     # 'log_file': 'HDFS/HDFS_1k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.2,
    #     'depth': 4,
    #     'tau': 0.7
    # },

    # 'HDFS100k': {
    #     'log_file': 'HDFS/HDFS_100k.log',
    #     # 'log_file': 'HDFS/HDFS_1k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.2,
    #     'depth': 4,
    #     'tau': 0.7
    # },

    # 'HDFS1m': {
    #     'log_file': 'HDFS/HDFS_1m.log',
    #     # 'log_file': 'HDFS/HDFS_1k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.2,
    #     'depth': 4,
    #     'tau': 0.7
    # },



    # 'HDFS': {
    #     'log_file': 'HDFS/HDFS_2k.log',
    #     # 'log_file': 'HDFS/HDFS_1k.log',

    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.2,
    #     'depth': 4,
    #     'tau': 0.7

    # },

    # 'Hadoop': {
    #     'log_file': 'Hadoop/Hadoop_2k.log',
    #     'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'st': 0.2,
    #     'depth': 4,
    #     'tau': 1

    # },

    # 'Spark': {
    #     'log_file': 'Spark/Spark_2k.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.1,
    #     'depth': 4,
    #     # 'tau': 0.55
    #     'tau': 0.7
    # },

    # 'Zookeeper': {
    #     'log_file': 'Zookeeper/Zookeeper_2k.log',
    #     'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
    #     'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.6,
    #     'depth': 4,
    #     # 'tau': 0.7
    #     'tau': 0.8
    # },

    # 'BGL': {
    #     'log_file': 'BGL/BGL_2k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4,
    #     # 'tau': 0.75
    #     'tau': 1.1
    # },

    # 'HPC': {
    #     'log_file': 'HPC/HPC_2k.log',
    #     'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
    #     'regex': [r'=\d+'],
    #     'st': 0.6,
    #     'depth': 4,
    #     # 'tau': 0.65
    #     'tau': 2.0
    # },

    # 'Thunderbird': {
    #     'log_file': 'Thunderbird/Thunderbird_2k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'st': 0.5,
    #     'depth': 4,
    #     # 'tau': 0.5
    #     'tau': 0.85

    # },

    # 'Windows': {
    #     'log_file': 'Windows/Windows_2k.log',
    #     'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
    #     'regex': [r'0x.*?\s'],
    #     'st': 0.8,
    #     'depth': 5,
    #     # 'tau': 0.7
    #     'tau': 0.75
    # },

    # 'Linux': {
    #     'log_file': 'Linux/Linux_2k.log',
    #     'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
    #     'st': 0.25,
    #     'depth': 6,
    #     # 'tau': 0.55
    #     'tau': 0.45

    # },

    # 'Andriod': {
    #     'log_file': 'Andriod/Andriod_2k.log',
    #     'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
    #     'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
    #     'st': 0.95,
    #     'depth': 6,
    #     'tau': 0.95
    # },

    # 'HealthApp': {
    #     'log_file': 'HealthApp/HealthApp_2k.log',
    #     'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
    #     'regex': [],
    #     'st': 0.1,
    #     'depth': 4,
    #     # 'tau': 0.8
    #     'tau': 2.0
    # },

    # 'Apache': {
    #     'log_file': 'Apache/Apache_2k.log',
    #     'log_format': '\[<Time>\] \[<Level>\] <Content>',
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'st': 0.0,
    #     'depth': 4,
    #     # 'tau': 0.6
    #     'tau': 1.0
    # },

    # 'Proxifier': {
    #     'log_file': 'Proxifier/Proxifier_2k.log',
    #     'log_format': '\[<Time>\] <Program> - <Content>',
    #     'regex': [r'<\d+\s?sec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
    #     'st': 0.6,
    #     'depth': 3,
    #     # 'tau': 0.85
    #     'tau': 1.0
    # },

    # 'OpenSSH': {
    #     'log_file': 'OpenSSH/OpenSSH_2k.log',
    #     'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    #     'regex': [r'([\w-]+\.){2,}[\w-]+', r'(\d+\.){3}\d+'],
    #     'st': 0.6,
    #     'depth': 5,
    #     # 'tau': 0.8
    #     'tau': 1.1
    # },

    # 'OpenStack': {
    #     'log_file': 'OpenStack/OpenStack_2k.log',
    #     'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
    #     'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\s\d+\s'],
    #     'st': 0.7,
    #     'depth': 5,
    #     # 'tau': 0.9
    #     'tau': 0.9
    # },

    # 'Mac': {
    #     'log_file': 'Mac/Mac_2k.log',
    #     'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
    #     'regex': [r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.6,
    #     'depth': 6,
    #     # 'tau': 0.6
    #     'tau': 0.89
    # },
}

#AAAA

parser = argparse.ArgumentParser()
parser.add_argument('--dictionary', default='EngCorpus.pkl', type=str)
args = parser.parse_args()
corpus = args.dictionary

bechmark_result = []
for dataset, setting in benchmark_settings.items():
    print('\n=== Evaluation on %s ===' % dataset)
    indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
    log_file = os.path.basename(setting['log_file'])
    
    headers, regex = generate_logformat_regex(setting['log_format'])
    log_messages = load_logs(input_dir + setting['log_file'], regex, headers)

    start_time = datetime.now()


    log_messages = KnowledgeGroupLayer(log_messages).run()
    log_messages = TokenizeGroupLayer(log_messages, rex=setting['regex']).run()
    
    log_messages = DictGroupLayer(log_messages, corpus).run()


    best_st = -1
    best_tau = -1
    best_acc = -1
    time_u = 0
    for var in range(0, 1):

        best_tau = setting['tau']
        best_st = setting['st'] 

        # tau_tmp = var /10
        # st_tmp = setting['st']

        # tau_tmp = setting['tau']
        # st_tmp = var /10

        tau_tmp = setting['tau']
        st_tmp = setting['st']
        print("st_tmp={} tau_tmp={}".format(st_tmp , tau_tmp))
        parser = Drain_plus_2.LogParser(log_messages, log_format=setting['log_format'], indir=indir, outdir=output_dir, rex=setting['regex'],
                        depth=setting['depth'], st=st_tmp,tau=tau_tmp)

        # parser = Drain_plus_2.LogParser(log_messages, log_format=setting['log_format'], indir=indir, outdir=output_dir, rex=setting['regex'],
        #                         depth=setting['depth'], st=setting['st'],tau=setting['tau'])
        parser.parse(log_file)
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))


        F1_measure, accuracy = evaluator.evaluate(
            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
            parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
        )

        if accuracy > best_acc:
            best_acc = accuracy
            best_st = st_tmp
            best_tau = tau_tmp

    # bechmark_result.append([dataset, best_st, best_tau , F1_measure, best_acc, time_u.seconds/100.0])
    bechmark_result.append([dataset, best_st, best_tau , F1_measure, best_acc])

    print("best_st={}, best_tau={} , F1_measure={}, best_acc={} , ".format(best_st, best_tau , F1_measure, best_acc ))

print('\n=== Overall evaluation results ===')
df_result = pd.DataFrame(bechmark_result, columns=['Dataset','best_st','best_tau', 'F1_measure', 'Accuracy' ])
df_result.set_index('Dataset', inplace=True)
print(df_result)
df_result.T.to_csv('Drain_plus_2_bechmark_result.csv')


avg_accr = 0
for i in range(len(bechmark_result)):
    avg_accr += bechmark_result[i][4]
avg_accr /= len(bechmark_result)
print('avarage accuracy is {}'.format(avg_accr))

