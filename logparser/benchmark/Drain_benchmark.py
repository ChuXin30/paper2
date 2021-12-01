#!/usr/bin/env python

import sys
sys.path.append('../')
from logparser import  Drain, evaluator
import os
import pandas as pd
import time
import datetime


input_dir = '../logs/' # The input directory of log file
output_dir = 'Drain_result/' # The output directory of parsing results

benchmark_settings = {

    # 'BGL1k': {
    #     'log_file': 'BGL/BGL_1k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'BGL10k': {
    #     'log_file': 'BGL/BGL_10k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'BGL100k': {
    #     'log_file': 'BGL/BGL_100k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'BGL1m': {
    #     'log_file': 'BGL/BGL_1m.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },
    # 'HDFS1k': {
    #     'log_file': 'HDFS/HDFS_1k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'HDFS10k': {
    #     'log_file': 'HDFS/HDFS_10k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'HDFS100k': {
    #     'log_file': 'HDFS/HDFS_100k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'HDFS1m': {
    #     'log_file': 'HDFS/HDFS_1m.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'Spark1m': {
    #     'log_file': 'Spark/Spark_1m_match.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.5,
    #     'depth': 4
    #     },


    # 'Spark1k': {
    #     'log_file': 'Spark/Spark_1k_match.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'Spark10k': {
    #     'log_file': 'Spark/Spark_10k_match.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'Spark100k': {
    #     'log_file': 'Spark/Spark_100k_match.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.5,
    #     'depth': 4
    #     },


        'HDFS1_all': {
        'log_file': 'HDFS/HDFS.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'depth': 4
        },

    # 'OpenSSH600': {
    #     'log_file': 'OpenSSH/OpenSSH_600_match.log',
    #     'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.6,
    #     'depth': 5   
    #     },

    # 'OpenSSH6K': {
    #     'log_file': 'OpenSSH/OpenSSH_6k_match.log',
    #     'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.6,
    #     'depth': 5   
    #     },

    # 'OpenSSH60K': {
    #     'log_file': 'OpenSSH/OpenSSH_60k_match.log',
    #     'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.6,
    #     'depth': 5   
    #     },

    # 'OpenSSH600k': {
    #     'log_file': 'OpenSSH/OpenSSH_600k_match.log',
    #     'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.6,
    #     'depth': 5   
    #     },

    # 'OpenSSH2k': {
    #     'log_file': 'OpenSSH/OpenSSH_2k.log',
    #     'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.6,
    #     'depth': 5   
    #     },

    # 'Andriod': {
    #     'log_file': 'Andriod/Andriod.log',
    #     'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
    #     'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
    #     'st': 0.2,
    #     'depth': 6   
    #     },

    # 'BGL100': {
    #     'log_file': 'BGL/BGL_100.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },



    # 'BGL400': {
    #     'log_file': 'BGL/BGL_400.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'BGL4k': {
    #     'log_file': 'BGL/BGL_4k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'BGL40k': {
    #     'log_file': 'BGL/BGL_40k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'BGL400k': {
    #     'log_file': 'BGL/BGL_400k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },





    # 'BGL1m': {
    #     'log_file': 'BGL/BGL_1m.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'Spark100': {
    #     'log_file': 'Spark/Spark_100.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'HDFS': {
    #     'log_file': 'HDFS/HDFS_10m.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'HDFS': {
    #     'log_file': 'HDFS/HDFS_2k.log',
    #     # 'log_file': 'HDFS/HDFS_1k.log',

    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'Hadoop': {
    #     'log_file': 'Hadoop/Hadoop_2k.log',
    #     'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>', 
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'Spark': {
    #     'log_file': 'Spark/Spark_2k.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'Zookeeper': {
    #     'log_file': 'Zookeeper/Zookeeper_2k.log',
    #     'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
    #     'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'BGL': {
    #     'log_file': 'BGL/BGL_2k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'HPC': {
    #     'log_file': 'HPC/HPC_2k.log',
    #     'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
    #     'regex': [r'=\d+'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'Thunderbird': {
    #     'log_file': 'Thunderbird/Thunderbird_2k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'Windows': {
    #     'log_file': 'Windows/Windows_2k.log',
    #     'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
    #     'regex': [r'0x.*?\s'],
    #     'st': 0.7,
    #     'depth': 5      
    #     },

    # 'Linux': {
    #     'log_file': 'Linux/Linux_2k.log',
    #     'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
    #     'st': 0.39,
    #     'depth': 6        
    #     },

    # 'Andriod': {
    #     'log_file': 'Andriod/Andriod_2k.log',
    #     'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
    #     'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
    #     'st': 0.2,
    #     'depth': 6   
    #     },

    # 'HealthApp': {
    #     'log_file': 'HealthApp/HealthApp_2k.log',
    #     'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
    #     'regex': [],
    #     'st': 0.2,
    #     'depth': 4
    #     },

    # 'Apache': {
    #     'log_file': 'Apache/Apache_2k.log',
    #     'log_format': '\[<Time>\] \[<Level>\] <Content>',
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'Proxifier': {
    #     'log_file': 'Proxifier/Proxifier_2k.log',
    #     'log_format': '\[<Time>\] <Program> - <Content>',
    #     'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
    #     'st': 0.6,
    #     'depth': 3
    #     },

    # 'OpenSSH': {
    #     'log_file': 'OpenSSH/OpenSSH_2k.log',
    #     'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.6,
    #     'depth': 5   
    #     },

    # 'OpenStack': {
    #     'log_file': 'OpenStack/OpenStack_2k.log',
    #     'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
    #     'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
    #     'st': 0.5,
    #     'depth': 5
    #     },

    # 'Mac': {
    #     'log_file': 'Mac/Mac_2k.log',
    #     'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
    #     'regex': [r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.7,
    #     'depth': 6   
    #     },
}

bechmark_result = []
for dataset, setting in benchmark_settings.items():
    print('\n=== Evaluation on %s ==='%dataset)

    indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
    log_file = os.path.basename(setting['log_file'])

    starttime = datetime.datetime.now()
    parser = Drain.LogParser(log_format=setting['log_format'], indir=indir, outdir=output_dir, rex=setting['regex'], depth=setting['depth'], st=setting['st'])
    parser.parse(log_file)
    time_use = datetime.datetime.now() - starttime
    print("time_use = {}".format(time_use.seconds))

    F1_measure, accuracy = evaluator.evaluate(
                           groundtruth=os.path.join(indir, log_file + '_structured.csv'),
                           parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
                           )
    bechmark_result.append([dataset, F1_measure, accuracy ,   time_use.microseconds/1000000.0  ])

    # bechmark_result.append([dataset, F1_measure, accuracy ,   time_use.seconds  ])

avg_accr = 0
for i in range(len(bechmark_result)):
    avg_accr += bechmark_result[i][2]
avg_accr /= len(bechmark_result)
pd_result = pd.DataFrame(bechmark_result, columns={'dataset', 'F1_measure', 'Accuracy' , 'time'})
print(pd_result)
print('avarage accuracy is {}'.format(avg_accr))

print('\n=== Overall evaluation results ===')
df_result = pd.DataFrame(bechmark_result, columns=['Dataset', 'F1_measure', 'Accuracy' , 'time'])
df_result.set_index('Dataset', inplace=True)
print(df_result)
df_result.T.to_csv('Drain_bechmark_result.csv')
print(os.path.join(indir, log_file + '_structured.csv'))
avg_accr = 0
for i in range(len(bechmark_result)):
    avg_accr += bechmark_result[i][2]
avg_accr /= len(bechmark_result)
print('avarage accuracy is {}'.format(avg_accr))