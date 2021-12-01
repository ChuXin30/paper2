#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys
sys.path.append('../')
from logparser import Drain
if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # input_dir ='/media/hao/file/paper2_code/BGL/noise_2/'
    # output_dir = '/media/hao/file/paper2_code/BGL/noise_2/'  # The output directory of parsing results

    log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Content>'
    log_file   = 'BGL.log'
    regex      = [r'core\.\d+']
    st         = 0.6  # Similarity threshold
    depth      = 4  # Depth of all leaf nodes
    parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
    parser.parse(log_file)
#BGL noise
'''
input_dir ='/media/hao/file/paper2_code/BGL/noise/'
output_dir = '/media/hao/file/paper2_code/BGL/noise/'  # The output directory of parsing results
log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Content>'
log_file   = 'BGL.log'
regex      = [r'core\.\d+']
st         = 0.5  # Similarity threshold
depth      = 4  # Depth of all leaf nodes
parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)
'''
#noise_1
'''
input_dir ='/media/hao/file/paper2_code/BGL/noise/'
output_dir = '/media/hao/file/paper2_code/BGL/noise_1/'  # The output directory of parsing results
log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Content>'
log_file   = 'BGL.log'
regex      = [r'core\.\d+']
st         = 0.6  # Similarity threshold
depth      = 4  # Depth of all leaf nodes
parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)
'''
#noise_2
# input_dir ='/media/hao/file/paper2_code/BGL/noise_2/'
# output_dir = '/media/hao/file/paper2_code/BGL/noise_2/'  # The output directory of parsing results
# log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Content>'
# log_file   = 'BGL.log'
# regex      = [r'core\.\d+']
# st         = 0.6  # Similarity threshold
# depth      = 4  # Depth of all leaf nodes
# parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
# parser.parse(log_file)
#test bgl
'''
input_dir  = '../logs/BGL/'  # The input directory of log file
output_dir = 'Drain_result/'  # The output directory of parsing results
log_file   = 'BGL_2k.log'  # The input log file name

log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Content>'  # HDFS log format
# Regular expression list for optional preprocessing (default: [])
regex      = [r'core\.\d+']
st         = 0.5  # Similarity threshold
depth      = 4  # Depth of all leaf nodes
parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)
'''

#HDFS
'''
output_dir = 'Drain_result1'  # The output directory of parsing results
input_dir  = '../logs/HDFS/'  # The input directory of log file
print(output_dir)
#log_file   = 'HDFS.log'  # The input log file name
#log_file   = 'HDFS_noise.log'
log_file   = 'HDFS_2k.log'
# Regular expression list for optional preprocessing (default: [])
#input_dir ='/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_10/'
#output_dir = '/home/hao/Desktop/论文２　实验代码/DeepLog-master/data_noise/noise_10/'  # The output directory of parsing results
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
regex      = [
    r'blk_(|-)[0-9]+' , # block id
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
]
'''
#test
'''
output_dir = 'Drain_result1'  # The output directory of parsing results
input_dir  = '../logs/BGL/'  # The input directory of log file
log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Content>'
log_file = 'BGL_2k.log'
regex      = [
    r'core\.\d+'
]
st         = 0.5  # Similarity threshold
depth      = 4  # Depth of all leaf nodes
parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)
'''
#BGL
'''
input_dir ='/home/hao/Desktop/论文２　实验代码/BGL/data/'
output_dir = '/home/hao/Desktop/论文２　实验代码/BGL/bgl_dataset/drain_without_06_4'  # The output directory of parsing results
#log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Content>'
log_file   = 'BGL.log'
regex      = [
    r'core\.\d+'
]
#st         = 0.5  # Similarity threshold
st         = 0.6  # Similarity threshold

depth      = 4  # Depth of all leaf nodes
parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)
'''




'''
'BGL': {
    'log_file': 'BGL/BGL_2k.log',
    'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    'regex': [r'core\.\d+'],
    'st': 0.5,
    'depth': 4
},
'''

