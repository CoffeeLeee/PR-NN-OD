import os
import argparse
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import sys
import datetime
import csv

parser = argparse.ArgumentParser()

# learning parameters
parser.add_argument('-learning_rate', type = float, default=0.001)
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-num_epoch', type=int, default=100)
parser.add_argument('-epoch_start', type=int, default=0)
parser.add_argument('-num_batch', type=int, default=200)
parser.add_argument('-weight_decay', type=float, default=0.0001)
parser.add_argument('-eval_freq', type=int, default=1)
parser.add_argument('-eval_start', type=int, default=1)
parser.add_argument('-print_freq_ep', type=int, default=1)

parser.add_argument('-batch_size_snr_train', type=int, default=30)
parser.add_argument('-batch_size_snr_validate', type=int, default=600)

parser.add_argument('-prob_start', type=float, default=0.1)
parser.add_argument('-prob_up', type=float, default=0.01)
parser.add_argument('-prob_step_ep', type=int, default=50)

# storing path
parser.add_argument('-result', type=str, default='result.txt')
parser.add_argument('-checkpoint', type=str, default='./checkpoint_3.pth.tar')
parser.add_argument('-resume', default='', type=str, metavar='PATH', 
                    help='path to latest checkpoint (default:none)')

# PR-NN parameters
parser.add_argument('-eval_info_length', type=int, default=10000)
parser.add_argument('-dummy_length_start', type=int, default=5)
parser.add_argument('-dummy_length_end', type=int, default=5)
parser.add_argument('-eval_length', type=int, default=10)
parser.add_argument('-overlap_length', type=int, default=20)

# RNN parameters
parser.add_argument('-input_size', type=int, default=5)
parser.add_argument('-rnn_input_size', type=int, default=5)
parser.add_argument('-rnn_hidden_size', type=int, default=50)
parser.add_argument('-output_size', type=int, default=1)
parser.add_argument('-rnn_layer', type=int, default=4)
parser.add_argument('-rnn_dropout_ratio', type=float, default=0)

# channel parameters
parser.add_argument('-snr_start', type=float, default=9.99)
parser.add_argument('-snr_stop', type=float, default=10.03)
parser.add_argument('-snr_step', type=float, default=0.01)

def main():
    csv_tx_path = "PR-NN_Detector_PRchannel/individual/tx_waveform001.csv"
    csv_data_path = 'PR-NN_Detector_PRchannel/individual/test_data_snr10.csv'

    checkpoint_path = 'PR-NN_Detector_PRchannel/checkpoint_2.pth.tar'
    tx_frame = np.loadtxt(csv_tx_path,dtype=np.float32,delimiter=',')
    data_frame = np.loadtxt(csv_data_path,dtype=np.float32,delimiter=',')

    global args
    args = parser.parse_known_args()[0]
    device = "cuda"
    model = Model(args, device).to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    



class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        
        '''
        time_step: total number of time steps in RNN
        fc_length: input length to linear layer
        dec_input: input linear network
        dec_rnn: rnn network
        dec_output: output linear network
        '''
        
        self.args = args
        self.device = device
        self.time_step = (args.dummy_length_start + args.eval_length 
                          + args.overlap_length + args.dummy_length_end)
        self.fc_length = args.eval_length + args.overlap_length
        self.dec_input = torch.nn.Linear(args.input_size, 
                                         args.rnn_input_size)
        self.dec_rnn = torch.nn.GRU(args.rnn_input_size, 
                                    args.rnn_hidden_size, 
                                    args.rnn_layer, 
                                    bias=True, 
                                    batch_first=True,
                                    dropout=args.rnn_dropout_ratio, 
                                    bidirectional=True)
        
        self.dec_output = torch.nn.Linear(2*args.rnn_hidden_size, args.output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        dec = torch.zeros(batch_size, self.fc_length, 
                          args.output_size).to(self.device)
        
        x = self.dec_input(x)
        y, _  = self.dec_rnn(x)
        y_dec = y[:, args.dummy_length_start : 
                  self.time_step-args.dummy_length_end, :]

        dec = torch.sigmoid(self.dec_output(y_dec))
        
        return torch.squeeze(dec, 2)