#!/usr/bin/env python
import sys
sys.path.append('../')
from logparser import SHISO

# input_dir   = '../logs/HDFS/' # The input directory of log file
# output_dir  = 'SHISO_result/' # The output directory of parsing results
# log_file    = 'HDFS_2k.log' # The input log file name

print("\n------------SHISO---------------")
input_dir = sys.argv[1]
output_dir = sys.argv[2]
log_file = sys.argv[3]

#hdfs dataset
log_format  = '<Date> <Time> <Pid> <Level> <Component>: <Content>' # HDFS log format
regex       = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'] # Regular expression list for optional preprocessing (default: [])
maxChildNum = 4 # The maximum number of children for each internal node
mergeThreshold = 0.1 # Threshold for searching the most similar template in the children
formatLookupThreshold = 0.3 # Lowerbound to find the most similar node to adjust
superFormatThreshold  = 0.85 # Threshold of average LCS length, determing whether or not to create a super format

# BGL dataset
# log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
# regex = [r'core\.\d+']
# maxChildNum = 4 # The maximum number of children for each internal node
# mergeThreshold = 0.005 # Threshold for searching the most similar template in the children
# formatLookupThreshold = 0.3 # Lowerbound to find the most similar node to adjust
# superFormatThreshold  = 0.85 # Threshold of average LCS length, determing whether or not to create a super format


# #HPC dataset
# log_format = '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>'
# regex = [r'=\d+']
# maxChildNum = 3 # The maximum number of children for each internal node
# mergeThreshold = 0.003 # Threshold for searching the most similar template in the children
# formatLookupThreshold = 0.6 # Lowerbound to find the most similar node to adjust
# superFormatThreshold  = 0.4 # Threshold of average LCS length, determing whether or not to create a super format


# # Zookeeper dataset
# log_format = '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>'
# regex = [r'(/|)(\d+\.){3}\d+(:\d+)?']
# maxChildNum = 4 # The maximum number of children for each internal node
# mergeThreshold = 0.003 # Threshold for searching the most similar template in the children
# formatLookupThreshold = 0.3 # Lowerbound to find the most similar node to adjust
# superFormatThreshold  = 0.85 # Threshold of average LCS length, determing whether or not to create a super format


# # Proxifier dataset
# log_format = '\[<Time>\] <Program> - <Content>'
# regex = [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B']
# maxChildNum = 4 # The maximum number of children for each internal node
# mergeThreshold = 0.002 # Threshold for searching the most similar template in the children
# formatLookupThreshold = 0.3 # Lowerbound to find the most similar node to adjust
# superFormatThreshold  = 0.85 # Threshold of average LCS length, determing whether or not to create a super format



parser = SHISO.LogParser(log_format,indir=input_dir,outdir=output_dir, rex=regex, maxChildNum=maxChildNum, 
                         mergeThreshold=mergeThreshold, formatLookupThreshold=formatLookupThreshold, 
                         superFormatThreshold=superFormatThreshold)
parser.parse(log_file)
