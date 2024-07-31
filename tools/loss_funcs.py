import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class MyMseMae(nn.Module):
    def __init__(self, w_mae=1, w_mse=1):
        super(MyMseMae, self).__init__()
        self.w_mae = w_mae
        self.w_mse = w_mse

    def forward(self, pred, gt):
        mse = torch.square(pred - gt)
        mae = torch.abs(pred - gt)
        return (self.w_mse*torch.mean(mse) + self.w_mae*torch.mean(mae))

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, pred, gt):
        return torch.sqrt(self.criterion(pred, gt))

def direction_loss(y1, y2):
    """
    y1, y2: tensors of shape (B, S, C)
    """
    epsilon = 1e-8
    y1_normalized = F.normalize(y1 + epsilon, dim=2, eps=1e-8)
    y2_normalized = F.normalize(y2 + epsilon, dim=2, eps=1e-8)

    cosine_similarity = torch.sum(y1_normalized * y2_normalized + epsilon, dim=2)

    # Add a small epsilon to handle the case when y1 and y2 are exactly the same
    #epsilon = 1e-8
    cosine_similarity = cosine_similarity.clamp(-1 + epsilon, 1 - epsilon)

    angle_difference = torch.acos(cosine_similarity)
    #print(torch.mean(angle_difference))
    return torch.mean(angle_difference)
    
def euclidean_distance_loss(y1, y2):
    """
    y1, y2: tensors of shape (B, S, C)
    """

    # Compute element-wise squared difference
    squared_diff = (y1 - y2) ** 2

    # Sum along the channel dimension (C)
    squared_diff_sum = torch.sum(squared_diff, dim=2)

    # Compute the Euclidean distance by taking the square root
    euclidean_distance = torch.sqrt(squared_diff_sum)

    # Compute the mean distance across the batch and sequence dimensions
    loss = euclidean_distance.mean()

    return loss
          
class RMSE_direction(nn.Module):
    def __init__(self):
        super(RMSE_direction, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, pred, gt):
        print(pred, gt)
        #return torch.sqrt(self.criterion(pred, gt)) + direction_loss(pred, gt)
        return direction_loss(pred, gt) 

def ade(y1, y2):
    """
    y1, y2: (B, S, C)
    """

    loss = y1 - y2
    loss = loss ** 2
    loss = np.sqrt(np.sum(loss, axis=2))

    return np.mean(loss)


def fde(y1, y2):
    loss = (y1[:, -1, :] - y2[:, -1, :]) ** 2
    return np.mean(np.sqrt(np.sum(loss, axis=1)))
        