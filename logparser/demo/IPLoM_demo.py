#!/usr/bin/env python

import sys
sys.path.append('../')
from logparser import IPLoM
print("\n------------LPLoM---------------")
input_dir = sys.argv[1]
output_dir = sys.argv[2]
log_file = sys.argv[3]
# input_dir    = '../logs/HDFS/'  # The input directory of log file
# output_dir   = 'IPLoM_result/'  # The output directory of parsing results
# log_file     = 'HDFS_2k.log'  # The input log file name


maxEventLen  = 120  # The maximal token number of log messages (default: 200)
step2Support = 0  # The minimal support for creating a new partition (default: 0)
upperBound   = 0.9  # The upper bound distance (default: 0.9)


CT           = 0.35  # The cluster goodness threshold (default: 0.35)
lowerBound   = 0.25  # The lower bound distance (default: 0.25)
log_format   = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
regex        = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']  # Regular expression list for optional preprocessing (default: [])

# #BGL dataset
# log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
# regex      = [r'core\.\d+']
# CT           = 0.4  # The cluster goodness threshold (default: 0.35)
# lowerBound   = 0.01  # The lower bound distance (default: 0.25)

# #HPC dataset
# log_format = '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>'
# regex = [r'=\d+']
# CT           = 0.58  # The cluster goodness threshold (default: 0.35)
# lowerBound   = 0.25  # The lower bound distance (default: 0.25)


# # Zookeeper dataset
# log_format = '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>'
# regex = [r'(/|)(\d+\.){3}\d+(:\d+)?']
# CT           = 0.4  # The cluster goodness threshold (default: 0.35)
# lowerBound   = 0.7  # The lower bound distance (default: 0.25)


# # Proxifier dataset
# log_format = '\[<Time>\] <Program> - <Content>'
# regex = [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B']
# CT           = 0.9 # The cluster goodness threshold (default: 0.35)
# lowerBound   = 0.25  # The lower bound distance (default: 0.25)

parser = IPLoM.LogParser(log_format=log_format, indir=input_dir, outdir=output_dir,
                         maxEventLen=maxEventLen, step2Support=step2Support, CT=CT, 
                         lowerBound=lowerBound, upperBound=upperBound, rex=regex)
parser.parse(log_file)
