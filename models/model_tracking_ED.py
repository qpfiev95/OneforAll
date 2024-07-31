from torch import nn
import torch
from models.make_layers import make_layers
from collections import OrderedDict

class Encoder(nn.Module):
    def __init__(self, subnets, rnns, time_step_orders=None):
        super().__init__()
        assert len(subnets)==len(rnns)
        self.time_step_orders = time_step_orders
        self.num_layer = len(subnets)
        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)
            
    def forward_by_timestep(self, input, output_list, state_list, mem_state_list):
        new_output_list, new_state_list, new_mem_state_list = [], [], []
        for i in range(1, self.num_layer+1):

            #print("----------", i, input.shape)
            #print(getattr(self, 'stage' + str(i)))
            input = getattr(self, 'stage' + str(i))(input.float())
            #print("----------", i, input.shape)
            #print(getattr(self, 'rnn' + str(i)))
            if len(state_list) == 0:
                output, new_state, mem_state = getattr(self, 'rnn' + str(i))(input, None, None) #################
            else:
                output, new_state, mem_state = getattr(self, 'rnn' + str(i))(input, state_list[i-1], mem_state_list[i-1])
            input = output
            
            new_output_list.append(output)
            new_state_list.append(new_state)
            new_mem_state_list.append(mem_state)
            
        return new_output_list, new_state_list, new_mem_state_list

    # inputs: B*S*C
    def forward(self, data_inputs):
        '''
        :param inputs: BxSxC
        :return:
        '''
        state_list = []
        mem_state_list = []
        output_list = []
        # At the first timestep: states = []
        output_list, state_list, mem_state_list = self.forward_by_timestep(data_inputs[:, 0, :], output_list, state_list, mem_state_list)
        #print(state.shape)
        for i in range(1, len(self.time_step_orders)):
            output_list, state_list, mem_state_list = self.forward_by_timestep(data_inputs[:, i, :], output_list, state_list, mem_state_list)
        #for i in range(len(output_list)):
        #    print("qqqqqqqqqqqqqqqqqqqq: ", output_list[i].shape, state_list[i].shape)
        #qqqqqqqqqqqqqqqqqqqq:  torch.Size([32, 64]) torch.Size([32, 64])
        #qqqqqqqqqqqqqqqqqqqq:  torch.Size([32, 128]) torch.Size([32, 128])

        return output_list, state_list, mem_state_list

class Decoder(nn.Module):
    def __init__(self, subnets, rnns, time_step_orders):
        super().__init__()
        self.time_step_orders = time_step_orders
        self.num_layer = len(rnns)
        for index, rnn in enumerate(rnns):
            setattr(self, 'rnn' + str(self.num_layer - index), rnn)
        for index, params in enumerate(subnets):
            setattr(self, 'stage' + str(self.num_layer - index + 1), make_layers(params))

    def forward_by_timestep(self, output_list, state_list, mem_state_list):
        new_output_list, new_state_list, new_mem_state_list = [], [], []
        for i in range(self.num_layer-1, -1, -1):  # 1 0
            #print("Layer: ", i)
            #print(getattr(self, 'stage' + str(i+1)))
            #print(output_list[i].shape, state_list[i].shape)
            if i == self.num_layer-1:
                output, state, mem_state = getattr(self, 'rnn' + str(i+1))(output_list[i], state_list[i], mem_state_list[i])
            else:
                output, state, mem_state = getattr(self, 'rnn' + str(i+1))(input, state_list[i], mem_state_list[i])
            input = getattr(self, 'stage' + str(i+2))(output)
            
            new_output_list.append(output)
            new_state_list.append(state)
            new_mem_state_list.append(mem_state)
        prediction = self.stage1(input)
        new_output_list.reverse()
        new_state_list.reverse()
        new_mem_state_list.reverse()
        return new_output_list, new_state_list, new_mem_state_list, prediction

    def forward(self, output_list, state_list, mem_state_list):
        '''
        :param states_list: list of K-1 tensor 1xBxCxHxW
        :param states_list: list of K tensor 1xBxCxHxW
        :return:
        '''
        outputs = []

        for i in range(len(self.time_step_orders)):
            output_list, state_list, mem_state_list, prediction = \
                self.forward_by_timestep(output_list, state_list, mem_state_list)
            outputs.append(prediction)
        #print(torch.stack(outputs, dim=1).shape)
        return torch.stack(outputs, dim=1)

class ED(nn.Module):
    def __init__(self, encoder, decoder, timestep_en, timestep_de):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, input):
        output, state, mem_state = self.encoder(input)
        output = self.decoder(output, state, mem_state)
        return output
        
        
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
        #print("E:", l, c1, c2)
        ###
        E_connect_blocks.append(OrderedDict({f'linear_{activate}': [c1, c2]}))
        ###
        if block == "rnn":
            from models.components.vanilla_rnn import VanillaRNNCell
            E_central_blocks.append(VanillaRNNCell(input_channel=c2, hidden_channel=c2))
        elif block == "gru":
            from models.components.gru import GRUCell
            E_central_blocks.append(GRUCell(input_channel=c2, hidden_channel=c2))
        elif block == "lstm":
            from models.components.lstm import LSTMCell
            E_central_blocks.append(LSTMCell(input_channel=c2, hidden_channel=c2))
        else:
            raise "This block is not implemented!"
          
        
    for l in range(num_layer-1, -1, -1):
        ###
        print("D:", l, c1, c2)
        if block == "rnn":
            from models.components.vanilla_rnn import VanillaRNNCell
            D_central_blocks.append(VanillaRNNCell(input_channel=c2, hidden_channel=c2))
        elif block == "gru":
            from models.components.gru import GRUCell
            D_central_blocks.append(GRUCell(input_channel=c2, hidden_channel=c2))
        elif block == "lstm":
            from models.components.lstm import LSTMCell
            D_central_blocks.append(LSTMCell(input_channel=c2, hidden_channel=c2))
        else:
            raise "This block is not implemented!"
        ###
        D_connect_blocks.append(OrderedDict({f'linear_{activate}': [c2, c1]}))
        if l == 0:
            #c1, c2 = input_channel, hidden_channel  
            c2 = c1
            c1 = input_channel
            D_connect_blocks.append(OrderedDict({f'linear_{activate}': [c2, c1]}))
        else:
            #c1 = int(c1 / 2)
            c2 = c1
            #D_connect_blocks.append(OrderedDict({f'linear_{activate}': [c2, c1]})) 
            c1 = int(c2/2)
              
            
        
    
    E_nets = [E_connect_blocks, E_central_blocks]
    D_nets = [D_connect_blocks, D_central_blocks]
    return E_nets, D_nets
    

