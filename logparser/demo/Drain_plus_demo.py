#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys
sys.path.append('../')
from logparser import Drain_plus

if __name__ == '__main__':
    print("\n------------Drain_plus---------------")

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    log_file   = sys.argv[3]
    print (input_dir)
    print (output_dir)
    print (log_file)

    #HDFS data set
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    regex = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
    st         = 0.5  # Similarity threshold
    depth      = 4  # Depth of all leaf nodes

    tau =  0.6

    parser = Drain_plus.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex , tau = tau)
    parser.parse(log_file)

