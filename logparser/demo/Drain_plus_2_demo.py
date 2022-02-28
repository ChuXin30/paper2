#!/usr/bin/env python
# -*- coding: utf-8 -*

# import sys
# sys.path.append('../')
# from logparser import Drain_plus_2

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

if __name__ == '__main__':
    print("\n------------Drain_plus---------------")

    corpus = 'EngCorpus.pkl'

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    log_file   = sys.argv[3]
    tau   = float(sys.argv[4])
    depth = int(sys.argv[5])
    st = float(sys.argv[6])


    print (input_dir)
    print (output_dir)
    print (log_file)
    print (tau)

    #HDFS data set
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    regex = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
    # st         = 0.5  # Similarity threshold
    # depth      = 4  # Depth of all leaf nodes
    # tau =  0.6

    headers, regex1 = generate_logformat_regex(log_format)
    log_messages = load_logs(input_dir + log_file , regex1, headers)

    log_messages = KnowledgeGroupLayer(log_messages).run()
    log_messages = TokenizeGroupLayer(log_messages, rex=regex ).run()
    
    log_messages = DictGroupLayer(log_messages, corpus).run()


    # parser = Drain_plus_2.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex , tau = tau)
    # parser.parse(log_file)
    parser = Drain_plus_2.LogParser(log_messages, log_format, indir=input_dir, outdir=output_dir, rex=regex, depth = depth, st = st,tau = tau)
    parser.parse(log_file)

