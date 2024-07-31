import sys
sys.path.insert(0, '..')
from collections import OrderedDict


# parameters
'''
input_channel, hidden_channel = 3, 64

encoder_params = [
    [
# in_channel, out_channel, kernel_size, stride, padding
        OrderedDict({'linear_leaky': [input_channel, hidden_channel]})
    ],
    [
        GRUCell(input_channel=hidden_channel, hidden_channel=hidden_channel)
    ]
]
decoder_params = [
    [
        OrderedDict({
            'linear_leaky': [hidden_channel, input_channel]
        }),
    ],
    [
        GRUCell(input_channel=hidden_channel, hidden_channel=hidden_channel)
    ]
]
'''

def ED_conf(input_channel, hidden_channel, num_layer, block, activate="leaky"):
    E_connect_blocks = []
    E_central_blocks = []
    D_connect_blocks = []
    D_central_blocks = []
    for l in range(num_layer):
        if l == 0:   
            c1, c2 = input_channel, hidden_channel  
        else:
            c1 = c2
            c2 = c2 * 2  
        ###
        E_connect_blocks.append(OrderedDict({f'linear_{activate}': [c1, c2]}))
        ###
        if block == "rnn":
            from models.components.vanilla_rnn import VanillaRNNCell
            E_central_blocks.append(VanillaRNNCell(input_channel=c2, hidden_channel=c2))
        elif block == "gru"
            from models.components.gru import GRUCell
            E_central_blocks.append(GRUCell(input_channel=c2, hidden_channel=c2))
        elif block == "lstm"
            from models.components.lstm import LSTMCell
            E_central_blocks.append(LSTMCell(input_channel=c2, hidden_channel=c2))
        else:
            raise "This block is not implemented!"
          
        
    for l in range(num_layer-1:-1:-1):
        ###
        if block == "rnn":
            from models.components.vanilla_rnn import VanillaRNNCell
            E_central_blocks.append(VanillaRNNCell(input_channel=c2, hidden_channel=c2))
        elif block == "gru"
            from models.components.gru import GRUCell
            E_central_blocks.append(GRUCell(input_channel=c2, hidden_channel=c2))
        elif block == "lstm"
            from models.components.lstm import LSTMCell
            E_central_blocks.append(LSTMCell(input_channel=c2, hidden_channel=c2))
        else:
            raise "This block is not implemented!"
        ###
        if l == 0:
            c1, c2 = input_channel, hidden_channel  
        else:
            c1 = c2 / 2
            c2 = c1
        
        E_connect_blocks.append(OrderedDict({f'linear_{activate}': [c2, c1]}))
    
    E_nets = [E_connect_blocks, E_central_blocks]
    D_nets = [D_connect_blocks, D_central_blocks]
    return E_nets, D_nets
         
        
    
