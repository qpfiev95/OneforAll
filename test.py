import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal


class KalmanFilter(nn.Module):
    def __init__(self, input_size, state_size, output_size):
        super(KalmanFilter, self).__init__()

        # Kalman Filter parameters
        self.transition = nn.Linear(input_size, state_size)
        self.transition_cov = nn.Parameter(torch.eye(state_size))
        self.observation = nn.Linear(state_size, output_size)
        self.observation_cov = nn.Parameter(torch.eye(output_size))

        self.state_size = state_size

    def forward(self, x):
        batch_size, sequence_length, input_size = x.size()

        # Initialize initial state and covariance
        initial_state = self.transition(x[:, 0, :])
        initial_covariance = torch.eye(self.state_size).unsqueeze(0)

        filtered_states = [initial_state]
        filtered_covariances = [initial_covariance]

        # Kalman filter forward pass
        for t in range(1, sequence_length):
            # Prediction step
            predicted_state = self.transition(x[:, t, :])

            # Add noise to predicted state
            noise_dist = MultivariateNormal(torch.zeros(self.state_size), self.transition_cov)
            predicted_state += noise_dist.sample((batch_size,))

            predicted_covariance = torch.matmul(
                torch.matmul(self.transition_cov, filtered_covariances[-1]), self.transition_cov.t()
            )

            # Update step
            innovation = x[:, t, :] - self.observation(predicted_state)
            innovation_covariance = torch.matmul(
                torch.matmul(self.observation_cov, predicted_covariance), self.observation_cov.t()
            )
            kalman_gain = torch.matmul(
                torch.matmul(predicted_covariance, self.observation_cov.t()),
                torch.inverse(innovation_covariance.unsqueeze(-1)).squeeze(-1),
            )

            updated_state = predicted_state + torch.matmul(kalman_gain, innovation.unsqueeze(-1)).squeeze(-1)
            updated_covariance = predicted_covariance - torch.matmul(
                torch.matmul(kalman_gain, self.observation_cov), predicted_covariance
            )

            filtered_states.append(updated_state)
            filtered_covariances.append(updated_covariance)

        filtered_states = torch.stack(filtered_states, dim=1)
        return self.observation(filtered_states)


# Example usage
C = 3
S = 3
T = 3
input_size = C  # Dimension of input sequence
state_size = 64
output_size = C  # Dimension of output sequence

# Create an instance of KalmanFilter model
kalman_filter = KalmanFilter(input_size, state_size, output_size)

# Generate random input sequence
input_sequence = torch.randn(S, T, C)  # Shape: SxTxC

# Obtain predicted output sequence
predicted_output_sequence = kalman_filter(input_sequence)