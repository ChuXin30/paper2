# -*- coding: utf-8 -*-
#!/usr/bin/env python
from swiss_layers.file_output_layer import FileOutputLayer
from swiss_layers.knowledge_layer import KnowledgeGroupLayer
from swiss_layers.mask_layer import MaskLayer
from swiss_layers.tokenize_group_layer import TokenizeGroupLayer
from swiss_layers.dict_group_layer import DictGroupLayer


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

input_dir = '../logs/' # The input directory of log file
output_dir = 'SwissLogParserResult/' # The output directory of parsing results


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

    # 'BGL_all': {
    #     'log_file': 'BGL/BGL.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    # },

    'HDFS_all': {
        'log_file': 'HDFS/HDFS.log',
        # 'log_file': 'HDFS/HDFS_1k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    },

    # 'Spark100k': {
    #     'log_file': 'Spark/Spark_100k_match.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    # },

    # 'Spark10k': {
    #     'log_file': 'Spark/Spark_10k_match.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    # },

    # 'Spark1k': {
    #     'log_file': 'Spark/Spark_1k_match.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    # },

    # 'Spark1m': {
    #     'log_file': 'Spark/Spark_1m_match.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    # },



    # 'BGL1m': {
    #     'log_file': 'BGL/BGL_1m.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    # },

    # 'BGL1k': {
    #     'log_file': 'BGL/BGL_1k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    # },

    # 'BGL10k': {
    #     'log_file': 'BGL/BGL_10k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    # },

    # 'BGL100k': {
    #     'log_file': 'BGL/BGL_100k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    # },


    # 'HDFS1k': {
    #     'log_file': 'HDFS/HDFS_1k.log',
    #     # 'log_file': 'HDFS/HDFS_1k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    # },

    # 'HDFS10k': {
    #     'log_file': 'HDFS/HDFS_10k.log',
    #     # 'log_file': 'HDFS/HDFS_1k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    # },

    # 'HDFS100k': {
    #     'log_file': 'HDFS/HDFS_100k.log',
    #     # 'log_file': 'HDFS/HDFS_1k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    # },

    # 'HDFS1m': {
    #     'log_file': 'HDFS/HDFS_1m.log',
    #     # 'log_file': 'HDFS/HDFS_1k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    # },




    # 'HDFS': {
    #     'log_file': 'HDFS/HDFS_2k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
    #     },

    # 'Hadoop': {
    #     'log_file': 'Hadoop/Hadoop_2k.log',
    #     'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>', 
    #     'regex': [r'(\d+\.){3}\d+']
    #     },

    # 'Spark': {
    #     'log_file': 'Spark/Spark_2k.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+']
    #     },

    # 'Zookeeper': {
    #     'log_file': 'Zookeeper/Zookeeper_2k.log',
    #     'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
    #     'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?']
    #     },

    # 'BGL': {
    #     'log_file': 'BGL/BGL_2k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+']
    #     },

    # 'HPC': {
    #     'log_file': 'HPC/HPC_2k.log',
    #     'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
    #     'regex': [r'=\d+']
    #     },

    # 'Thunderbird': {
    #     'log_file': 'Thunderbird/Thunderbird_2k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
    #     'regex': [r'(\d+\.){3}\d+']
    #     },

    # 'Windows': {
    #     'log_file': 'Windows/Windows_2k.log',
    #     'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
    #     'regex': [r'0x.*?\s']
    #     },

    # 'Linux': {
    #     'log_file': 'Linux/Linux_2k.log',
    #     'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
    #     },

    # 'Andriod': {
    #     'log_file': 'Andriod/Andriod_2k.log',
    #     'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
    #     'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b']
    #     },

    # 'HealthApp': {
    #     'log_file': 'HealthApp/HealthApp_2k.log',
    #     'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
    #     'regex': []
    #     },

    # 'Apache': {
    #     'log_file': 'Apache/Apache_2k.log',
    #     'log_format': '\[<Time>\] \[<Level>\] <Content>',
    #     'regex': [r'(\d+\.){3}\d+']
    #     },

    # 'Proxifier': {
    #     'log_file': 'Proxifier/Proxifier_2k.log',
    #     'log_format': '\[<Time>\] <Program> - <Content>',
    #     'regex': [r'<\d+\s?sec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B']
    #     },


    # 'OpenSSH': {
    #     'log_file': 'OpenSSH/OpenSSH_2k.log',
    #     'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    #     'regex': [r'([\w-]+\.){2,}[\w-]+', r'(\d+\.){3}\d+']
    #     },

    # 'OpenStack': {
    #     'log_file': 'OpenStack/OpenStack_2k.log',
    #     'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
    #     'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\s\d+\s']
    #     },
        
    # 'Mac': {
    #     'log_file': 'Mac/Mac_2k.log',
    #     'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
    #     'regex': [r'([\w-]+\.){2,}[\w-]+']
    #     }
}

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dictionary', default='EngCorpus.pkl', type=str)
    args = parser.parse_args()
    corpus = args.dictionary

    benchmark_result = []
    for dataset, setting in benchmark_settings.items():
        print('\n=== Evaluation on %s ==='%dataset)
        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        outdir = os.path.join(output_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])
        start_time = datetime.now()

        filepath = os.path.join(indir, log_file)
        print('Parsing file: ' + filepath)
        starttime = datetime.now()
        headers, regex = generate_logformat_regex(setting['log_format'])
        log_messages = load_logs(filepath, regex, headers)
        # print(log_messages[1])
        # preprocess layer 这一步
        log_messages = KnowledgeGroupLayer(log_messages).run()
        # print(log_messages[1])
        # tokenize layer 没有这一步结果再avarage accuracy is 0.6954374999999999 更低了 那我原来的方法如果使用了这一步结果会不会更好呢？
        log_messages = TokenizeGroupLayer(log_messages, rex=setting['regex']).run()
        
        # for line_log in log_messages.items():
        #     print(line_log[1]['Content'])
        #     print(line_log[1]['LineId'])
        
        # dictionarize layer and cluster by wordset
        dict_group_result = DictGroupLayer(log_messages, corpus).run()
        # print(dict_group_result[1])
        # print(dict_group_result[2])
        # print(len(dict_group_result))
        

        # apply LCS and prefix tree
        results, templates = MaskLayer(dict_group_result).run()
        output_file = os.path.join(outdir, log_file)
        # output parsing results
        FileOutputLayer(log_messages, output_file, templates, ['LineId'] + headers).run()
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
        F1_measure, accuracy = evaluator.evaluate(
                            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
                            parsedresult=os.path.join(outdir, log_file + '_structured.csv')
                            )
        benchmark_result.append([dataset, F1_measure, accuracy])

    print('\n=== Overall evaluation results ===')
    avg_accr = 0
    for i in range(len(benchmark_result)):
        avg_accr += benchmark_result[i][2]
    avg_accr /= len(benchmark_result)
    pd_result = pd.DataFrame(benchmark_result, columns={'dataset', 'F1_measure', 'Accuracy'})
    print(pd_result)
    print('avarage accuracy is {}'.format(avg_accr))
    pd_result.to_csv('benchmark_result.csv', index=False)

