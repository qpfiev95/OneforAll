import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCell(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        super(GRUCell, self).__init__()

        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.update_gate = nn.Linear(input_channel + hidden_channel, hidden_channel)
        self.reset_gate = nn.Linear(input_channel + hidden_channel, hidden_channel)
        self.mem_gate = nn.Linear(input_channel + hidden_channel, hidden_channel)
        self.output_layer = nn.Linear(hidden_channel, input_channel)

    def forward(self, input, state, mem_state):
        if input is not None:
            batch_size = input.shape[0]
        else:
            batch_size = state.shape[0]
            input = torch.zeros(batch_size, self.hidden_channel)
            input = input.to(state.device)
        
        if state is not None:
            current_state = state
        else:
            current_state = torch.zeros(batch_size, self.hidden_channel)
            current_state = current_state.to(input.device)
        combined = torch.cat((input, current_state), dim=1)
        update_gate = F.sigmoid(self.update_gate(combined))
        reset_gate = F.sigmoid(self.reset_gate(combined))
        combined_state = torch.cat((input, current_state*reset_gate), dim=1)
        mem_gate = torch.tanh(self.mem_gate(combined_state))
        new_state = (1 - update_gate) * current_state + update_gate * mem_gate
        output = self.output_layer(new_state)
        mem_state = None
        return output, new_state, mem_state

### Testing
if __name__ == '__main__':
    input_channel, hidden_channel = 3, 64
    batch_size = 8

    # Create an instance of VanillaRNN model
    cell = GRUCell(input_channel, hidden_channel)

    # Generate random input sequence
    input = torch.randn(batch_size, input_channel)
    state = torch.randn(batch_size, hidden_channel)
    print(f"Input shape: {input.shape}")
    # Make prediction
    output, new_state, _ = cell(input, state)
    print(f"Output shape: {output.shape}, new state: {new_state.shape}")
