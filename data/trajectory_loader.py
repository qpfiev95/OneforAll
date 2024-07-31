import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TrajectoryLoader(Dataset):
    def __init__(self, file_path, input_len, output_len, irregular=False, offset=0):
        self.data = np.load(file_path)
        self.input_len = input_len
        self.output_len = output_len
        self.irregular = irregular
        self.offset = offset
        self.max_input_timestep = self.input_timestep_generator()
        self.max_index = [(i, j) for i in range(self.data.shape[0]) for j in range(self.max_input_timestep)]
        assert offset < input_len

    def input_timestep_generator(self):
        max_len = self.data.shape[1]
        max_len_input = max_len - self.output_len
        max_num_input = int(np.ceil(max_len_input // self.offset)) - 1
        return max_num_input

    def irr_generator(self):
        pass

    def __len__(self):
        return self.data.shape[0] * self.max_input_timestep

    def __getitem__(self, index):
        id = self.max_index[index]
        sample = self.data[id[0]]
        input_seq = sample[id[1]*self.offset:id[1]*self.offset + self.input_len]
        output_seq = sample[id[1]*self.offset + self.input_len : id[1]*self.offset + self.input_len + self.output_len]
        return input_seq, output_seq


### Testing
'''
file_path = "trajectory_data/p80_v5_a0.5_n10000_t50.npy"
data = np.load(file_path)
dataset = TrajectoryLoader(file_path=file_path, input_len=10, output_len=10, irregular=False, offset=5)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

# Iterate over the data loader
for input_seq, output_seq in dataloader:
    print(f"Input sequence shape: {input_seq.shape}, Output sequence shape: {output_seq.shape}")
'''
