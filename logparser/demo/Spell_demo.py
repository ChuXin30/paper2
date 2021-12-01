#!/usr/bin/env python
# -*- coding: utf-8 -*


import sys
sys.path.append('../')
from logparser import Spell

print("\n------------spell---------------")

input_dir = sys.argv[1]
output_dir = sys.argv[2]
log_file = sys.argv[3]

# input_dir = '../logs/HDFS/'  # The input directory of log file
# output_dir = 'IPLoM_result/'  # The output directory of parsing results
# log_file = 'HDFS_2k.log'  # The input log file name

# input_dir ='/media/hao/固态/paper2_code/DeepLog-master/data_noise/spell_noise_20/'
# output_dir = '/media/hao/固态/paper2_code/DeepLog-master/data_noise/spell_noise_20/'  # The output directory of parsing results
# log_file   = 'HDFS_noise.log'  # The input log file name

#HDFS dataset
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
tau        = 0.70  # Message type threshold (default: 0.5)
regex      = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']  # Regular expression list for optional preprocessing (default: [])

# #BGL dataset
# log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
# tau        = 0.75  # Message type threshold (default: 0.5)
# regex      = [r'core\.\d+']

# #HPC dataset
# log_format = '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>'
# regex = [r'=\d+']
# tau = 0.65


# # Zookeeper dataset
# log_format = '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>'
# regex = [r'(/|)(\d+\.){3}\d+(:\d+)?']
# tau = 0.7


# # Proxifier dataset
# log_format = '\[<Time>\] <Program> - <Content>'
# regex = [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B']
# tau = 0.85



parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
parser.parse(log_file)
