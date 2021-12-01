#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')

from logdeep.models.lstm import deeplog, loganomaly, robustlog
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *


# Config Parameters

options = dict()
options['data_dir'] = '../data/'
#options['window_size'] = 10
options['window_size'] = 20#loganomaly_bgl_drain_random_with_level_20

# options['device'] = "cpu"
options['device'] = "cuda"

# Smaple
options['sample'] = "sliding_window"
#options['window_size'] = 10  # if fix_window
options['window_size'] = 20  # if fix_window#loganomaly_bgl_drain_random_with_level_20

# Features
options['sequentials'] = True
options['quantitatives'] = True
options['semantics'] = False
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 1
#options['hidden_size'] = 64
options['hidden_size'] = 128

options['num_layers'] = 2
#options['num_classes'] = 28
#options['num_classes'] = 377
options['num_classes'] = 1915#loganomaly_bgl_drain_random_with_level_20

# Train
options['batch_size'] = 128
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 60
options['lr_step'] = (40, 60)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "loganomaly"
#options['save_dir'] = "../result/loganomaly/"
#options['save_dir'] = "../result/loganomaly_newtemplate/"
#options['save_dir'] = "../result/loganomaly_bgl/"
#options['save_dir'] = "../result/loganomaly_bgl_order_randown_20/"
# options['save_dir'] = "../result/loganomaly_bgl_drain_random_with_level_20/"
options['save_dir'] = "../result/loganomaly_bgl_noise_0/"

# Predict
#loganomaly
#options['model_path'] = "../result/loganomaly/loganomaly_epoch358.pth"#F1-measure: 97.296%
#options['model_path'] = "../result/loganomaly/loganomaly_bestloss.pth"#F1-measure: 97.247%
#options['model_path'] = "../result/loganomaly/loganomaly_epoch300.pth"#97.323%
#options['model_path'] = "../result/loganomaly/loganomaly_epoch306.pth"#97.564%
#options['model_path'] = "../result/loganomaly/loganomaly_epoch302.pth"

#loganomaly+new_template
#options['model_path'] = "../result/loganomaly_newtemplate/loganomaly_epoch302.pth"#F1-measure: 97.247%
#options['model_path'] = "../result/loganomaly_newtemplate/loganomaly_epoch308.pth"# 97.216%
#options['model_path'] = "../result/loganomaly_newtemplate/loganomaly_epoch324.pth"

#loganomaly_bgl_drain_random_with_level_20 window=20
#options['model_path'] = "../result/loganomaly_bgl_drain_random_with_level_20/loganomaly_epoch19.pth"
#options['num_candidates'] = 9#92.912%
#options['num_candidates'] = 8#91.347%
#options['num_candidates'] = 200#
#options['num_candidates'] = 250#


options['model_path'] = "../result/loganomaly_bgl_noise_0/loganomaly_last.pth"
options['num_candidates'] = 14#

seed_everything(seed=1234)


def train():
    Model = loganomaly(input_size=options['input_size'],
                       hidden_size=options['hidden_size'],
                       num_layers=options['num_layers'],
                       num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model = loganomaly(input_size=options['input_size'],
                       hidden_size=options['hidden_size'],
                       num_layers=options['num_layers'],
                       num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        predict()
