import torch
import torch.nn as nn


class VanillaRNNCell(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        super(VanillaRNNCell, self).__init__()

        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.hidden_layer = nn.Linear(input_channel + hidden_channel, hidden_channel)
        self.output_layer = nn.Linear(hidden_channel, input_channel)

    def forward(self, input, state, mem_state):
        batch_size = input.shape[0]
        if state is not None:
            current_state = state
        else:
            current_state = torch.zeros(batch_size, self.hidden_channel)
            current_state = current_state.to(input.device)
        combined = torch.cat((input, current_state), dim=1)
        new_state = torch.tanh(self.hidden_layer(combined))
        output = self.output_layer(new_state)
        mem_state = None
        return output, new_state, mem_state

### Testing
if __name__ == '__main__':
    input_channel, hidden_channel = 3, 64
    batch_size = 8

    # Create an instance of VanillaRNN model
    vanilla_rnn = VanillaRNNCell(input_channel, hidden_channel)

    # Generate random input sequence
    input = torch.randn(batch_size, input_channel)
    state = torch.randn(batch_size, hidden_channel)
    print(f"Input shape: {input.shape}")
    # Make prediction
    output, new_state, _ = vanilla_rnn(input, state)
    print(f"Output shape: {output.shape}, new state: {new_state.shape}")
