import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        super(LSTMCell, self).__init__()

        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.input_gate = nn.Linear(input_channel + hidden_channel, hidden_channel)
        self.forget_gate = nn.Linear(input_channel + hidden_channel, hidden_channel)
        self.mem_gate = nn.Linear(input_channel + hidden_channel, hidden_channel)
        self.output_gate = nn.Linear(input_channel + hidden_channel, hidden_channel)
        self.output_layer = nn.Linear(hidden_channel, input_channel)

    def forward(self, input, state, mem_state):
        batch_size = input.shape[0]
        if state is not None:
            current_state = state
        else:
            current_state = torch.zeros(batch_size, self.hidden_channel)
            current_state = current_state.to(input.device)
        if mem_state is not None:
            current_mem_state = mem_state
        else:
            current_mem_state = torch.zeros(batch_size, self.hidden_channel)
            current_mem_state = current_mem_state.to(input.device)
        combined = torch.cat((input, current_state), dim=1)
        input_gate = F.sigmoid(self.input_gate(combined))
        forget_gate = F.sigmoid(self.forget_gate(combined))
        output_gate = F.sigmoid(self.output_gate(combined))
        mem_gate = torch.tanh(self.mem_gate(combined))
        new_mem_state = forget_gate * current_mem_state + input_gate * mem_gate
        new_state = output_gate * torch.tanh(new_mem_state)
        output = self.output_layer(new_state)

        return output, new_state, new_mem_state

### Testing
if __name__ == '__main__':
    input_channel, hidden_channel = 3, 64
    batch_size = 8

    # Create an instance of VanillaRNN model
    cell = LSTMCell(input_channel, hidden_channel)

    # Generate random input sequence
    input = torch.randn(batch_size, input_channel)
    state = torch.randn(batch_size, hidden_channel)
    mem_state = torch.randn(batch_size, hidden_channel)
    print(f"Input shape: {input.shape}")
    # Make prediction
    output, new_state, new_mem_state = cell(input, state, mem_state)
    print(f"Output shape: {output.shape}, new state: {new_state.shape}, new mem state: {new_mem_state.shape}")
