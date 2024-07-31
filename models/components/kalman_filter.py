import torch
import torch.nn as nn


class KalmanFilter(nn.Module):
    def __init__(self, input_channel, state_channel, output_channel):
        super(KalmanFilter, self).__init__()

        # kalman Filter params
        self.transition = nn.Linear(input_channel, state_channel)
        self.transition_cov = nn.Parameter(torch.eye(state_channel))
        self.observation = nn.Linear(input_channel, state_channel)
        self.observation = nn.Parameter(torch.eye(state_channel))

        self.input_channel = input_channel
        self.state_channel = state_channel
        self.output_channel = output_channel

    def forward(self, x):
        # x: input
        b, s, c = x.size()

        # Init the initial state and covariance
        initial_state = self.transition(x[:, 0, :])
        initial_covariance = torch.eye(self.state_channel).unsqueeze(0)
        filtered_states = [initial_state]
        filtered_covariances = [initial_covariance]

        # Kalman filter forward
        for t in range(1, s):
            # Prediction step
            predicted_state = self.transition(x[:, t, :])
