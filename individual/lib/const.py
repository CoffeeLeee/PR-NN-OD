import torch
import numpy as np

# Constant in experiments
'''
Encoder: (1,7)-RLL
ISI channel: EEPR4 channel
Dummy value mechanism: starting dummy values and ending dummy values
'''
def Constant():
    # Constrained encoder: (1,7)-RLL constraint, 4 states, 4 error propagations
    # Encoder_Dict[a][b]: a stands for each state, b stands for (1 - input tags, 2 - output words, 3 - next state)
    encoder_dict = {
        1 : {
            'input' : np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'output' : np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]]),
            'next_state' : np.array([[1], [2], [3], [3]])
        },
        2 : {
            'input' : np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'output' : np.array([[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1]]),
            'next_state' : np.array([[1], [2], [3], [4]])
        },
        3 : {
            'input' : np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'output' : np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1]]),
            'next_state' : np.array([[1], [2], [3], [4]])
        },
        4 : {
            'input' : np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'output' : np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]),
            'next_state' : np.array([[1], [2], [3], [3]])
        }
    }
    
    # channel state machine: EEPR4 channel
    channel_dict = {
        'state_machine' : np.array([
            [0, 0], [0, 1], [1, 2], [2, 3], [2, 4], [3, 7], [4, 8], [4, 9], 
            [5, 0], [5, 1], [6, 2], [7, 5], [7, 6], [8, 7], [9, 8], [9, 9]
        ]),
        'in_out' : np.array([
            [0, 0], [1, 1], [1, 3], [0, 2], [1, 3], [0, -2], [0, 0], [1, 1], 
            [0, -1], [1, 0], [1, 2], [0, -3], [1, -2], [0, -3], [0, -1], [1, 0]
        ]),
        'state_label' : np.array([
            [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
            [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0],
            [1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]
        ]),
        'num_state' : 10,
        'ini_state' : 0
    }

    
    
    # normalize channel outputs
    channel_in_out_norm = np.zeros(channel_dict['in_out'].shape)
    channel_in_out_norm[:, 0] = channel_dict['in_out'][:, 0]
    channel_in_out_norm[:, 1] = channel_dict['in_out'][:, 1] / np.sqrt(10)
    channel_dict['in_out'] = channel_in_out_norm
    
    # List of starting dummy values (based on EEPR4 state machine)
    dummy_dict_start = {
        0 : torch.tensor([[0, 0, 0, 0, 0]] / np.sqrt(10)).float(), 
        1 : torch.tensor([[0, 0, 0, 0, 1]] / np.sqrt(10)).float(),
        2 : torch.tensor([[0, 0, 0, 1, 3]] / np.sqrt(10)).float(), 
        3 : torch.tensor([[0, 0, 1, 3, 2]] / np.sqrt(10)).float(),
        4 : torch.tensor([[0, 0, 1, 3, 3]] / np.sqrt(10)).float(), 
        5 : torch.tensor([[1, 3, 2, -2, -3]] / np.sqrt(10)).float(),
        6 : torch.tensor([[1, 3, 2, -2, -2]] / np.sqrt(10)).float(), 
        7 : torch.tensor([[0, 1, 3, 2, -2]] / np.sqrt(10)).float(),
        8 : torch.tensor([[0, 1, 3, 3, 0]] / np.sqrt(10)).float(), 
        9 : torch.tensor([[0, 1, 3, 3, 1]] / np.sqrt(10)).float()
    }
    
    # List of ending dummy values (based on EEPR4 state machine)
    dummy_dict_end = {
        0 : np.array([[0, 0, 0, 0, 0]] / np.sqrt(10)), 
        1 : np.array([[3, 2, -2, -3, -1]] / np.sqrt(10)),
        2 : np.array([[2, -2, -3, -1, 0]] / np.sqrt(10)), 
        3 : np.array([[-2, -3, -1, 0, 0]] / np.sqrt(10)),
        4 : np.array([[0, -3, -3, -1, 0]] / np.sqrt(10)), 
        5 : np.array([[-1, 0, 0, 0, 0]] / np.sqrt(10)), 
        6 : np.array([[2, 2, -2, -3, -1]] / np.sqrt(10)), 
        7 : np.array([[-3, -1, 0, 0, 0]] / np.sqrt(10)), 
        8 : np.array([[-3, -3, -1, 0, 0]] / np.sqrt(10)), 
        9 : np.array([[-1, -3, -3, -1, 0]] / np.sqrt(10)),
    }
    
    dummy_dict_end_eval = torch.tensor([[0, 0, 0, 0, 0]] / np.sqrt(10)).float()
    
    return (encoder_dict, channel_dict, dummy_dict_start, 
            dummy_dict_end, dummy_dict_end_eval)

def Constant_od():
    # Constrained encoder: (1,7)-RLL constraint, 4 states, 4 error propagations
    # Encoder_Dict[a][b]: a stands for each state, b stands for (1 - input tags, 2 - output words, 3 - next state)
    encoder_dict = {
        1 : {
            'input' : np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'output' : np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]]),
            'next_state' : np.array([[1], [2], [3], [3]])
        },
        2 : {
            'input' : np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'output' : np.array([[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1]]),
            'next_state' : np.array([[1], [2], [3], [4]])
        },
        3 : {
            'input' : np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'output' : np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1]]),
            'next_state' : np.array([[1], [2], [3], [4]])
        },
        4 : {
            'input' : np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'output' : np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]),
            'next_state' : np.array([[1], [2], [3], [3]])
        }
    }
    
    # channel state machine: EEPR4 channel
    channel_dict = {
        'state_machine' : np.array([
            [0, 0], [0, 1], [1, 2], [1, 3], [2, 4], [2, 5], [3, 6], [3, 7],
            [4, 8], [4, 9], [5, 10], [5, 11], [6, 12], [6, 13], [7, 14], [7, 15],
            [8, 0], [8, 1], [9, 2], [9, 3], [10, 4], [10, 5], [11, 6], [11, 7],
            [12, 8], [12, 9], [13, 10], [13, 11], [14, 12], [14, 13], [15, 14], [15, 15]
        ]),
        'in_out' : np.array([
            [-1, -8], [1, -6], [-1, -4], [1, -2], [-1, -4], [1, -2], [-1, 0], [1, 2],
            [-1, -4], [1, -2], [-1, 0], [1, 2], [-1, 0], [1, 2], [-1, 4], [1, 6],
            [-1, -6], [1, -4], [-1, -2], [1, 0], [-1, -2], [1, 0], [-1, 2], [1, 4],
            [-1, -2], [1, 0], [-1, 2], [1, 4], [-1, 2], [1, 4], [-1, 6], [1, 8]
        ]),
        'state_label' : np.array([
            [-1, -1, -1, -1],[-1, -1, -1, 1], [-1, -1, 1, -1],
            [-1, -1, 1, 1], [-1, 1, -1, -1], [-1, 1, -1, 1],
            [-1, 1, 1, -1], [-1, 1, 1, 1], [1, -1, -1, -1], 
            [1, -1, -1, 1], [1, -1, 1, -1], [1, -1, 1, 1],
            [1, 1, -1, -1], [1, 1, -1, 1], [1, 1, 1, -1], [1, 1, 1, 1]
        ]),
        'num_state' : 16,
        'ini_state' : 0
    }


    # normalize channel outputs
    channel_in_out_norm = np.zeros(channel_dict['in_out'].shape)
    channel_in_out_norm[:, 0] = channel_dict['in_out'][:, 0]
    channel_in_out_norm[:, 1] = channel_dict['in_out'][:, 1] / np.sqrt(65)
    channel_dict['in_out'] = channel_in_out_norm
    
    # List of starting dummy values (based on EEPR4 state machine)
    dummy_dict_start = {
        0 : torch.tensor([[-8, -8, -8, -8, -8]]) / np.sqrt(65),
        1 : torch.tensor([[-8, -8, -8, -8, -6]]) / np.sqrt(65),
        2 : torch.tensor([[-8, -8, -8, -6, -4]]) / np.sqrt(65),
        3 : torch.tensor([[-8, -8, -8, -6, -2]]) / np.sqrt(65) ,
        4 : torch.tensor([[-8, -8, -6, -4, -4]]) / np.sqrt(65) ,
        5 : torch.tensor([[-8, -8, -6, -4, -2]]) / np.sqrt(65) ,
        6 : torch.tensor([[-8, -8, -6, -2,  0]]) / np.sqrt(65) ,
        7 : torch.tensor([[-8, -8, -6, -2,  2]]) / np.sqrt(65) ,
        8 : torch.tensor([[-8, -6, -4, -4, -4]]) / np.sqrt(65) ,
        9 : torch.tensor([[-8, -6, -4, -4, -2]]) / np.sqrt(65) ,
        10 : torch.tensor([[-8,  0, -2, -4, -6]]) / np.sqrt(65) ,
        11 : torch.tensor([[-8,  2, -2, -4, -6]]) / np.sqrt(65) ,
        12 : torch.tensor([[-8,  0,  0, -2, -6]]) / np.sqrt(65) ,
        13 : torch.tensor([[-8,  2,  0, -2, -6]]) / np.sqrt(65) ,
        14 : torch.tensor([[-8,  4,  2, -2, -6]]) / np.sqrt(65) ,
        15 : torch.tensor([[-8, -6, -2,  2,  6]]) / np.sqrt(65) ,
    }

    # List of ending dummy values (based on EEPR4 state machine)
    dummy_dict_end = {
        0 : np.array([[-8, -8, -8, -8, -8]] / np.sqrt(65)), 
        1 : np.array([[-4, -4, -4, -6, -8]] / np.sqrt(65)),
        2 : np.array([[-4, -4, -6, -8, -8]] / np.sqrt(65)), 
        3 : np.array([[0, 0, -2, -6, -8]] / np.sqrt(65)),
        4 : np.array([[-4, -6, -8, -8, -8]] / np.sqrt(65)), 
        5 : np.array([[0, -2, -4, -6, -8]] / np.sqrt(65)), 
        6 : np.array([[0, -2, -6, -8, -8]] / np.sqrt(65)), 
        7 : np.array([[4, 2, -2, -6, -8]] / np.sqrt(65)), 
        8 : np.array([[-6, -8, -8, -8, -8]] / np.sqrt(65)), 
        9 : np.array([[-2, -4, -4, -6, -8]] / np.sqrt(65)),
        10 : np.array([[-2, -4, -6, -8, -8]] / np.sqrt(65)),
        11 : np.array([[2, 0, -2, -6, -8]] / np.sqrt(65)),
        12 : np.array([[-2, -6, -8, -8, -8]] / np.sqrt(65)),
        13 : np.array([[2, -2, -4, -6, -8]] / np.sqrt(65)),
        14 : np.array([[2, -4, -6, -8, -8]] / np.sqrt(65)),
        15 : np.array([[6, 2, -2, -6, -8]] / np.sqrt(65)),
    }

    dummy_dict_end_path = {
        0 : np.array([[-1, -1, -1, -1, -1]]),
        1 : np.array([[1, -1, -1, -1, -1]]),
        3 : np.array([[-1, -1, -1, -1, -1]]),
        6 : np.array([[-1, -1, -1, -1, -1]]),
        7 : np.array([[-1, -1, -1, -1, -1]]),
        8 : np.array([[-1, -1, -1, -1, -1]]),
        9 : np.array([[1, -1, -1, -1, -1]]),
        12 : np.array([[-1, -1, -1, -1, -1]]),
        14 : np.array([[-1, -1, -1, -1, -1]]),
        15 : np.array([[-1, -1, -1, -1, -1]]),
    }

    dummy_dict_end_eval = torch.tensor([[-8, -8, -8, -8, -8]] / np.sqrt(65))
    
    return (encoder_dict, channel_dict, dummy_dict_start, 
            dummy_dict_end, dummy_dict_end_path, dummy_dict_end_eval)
