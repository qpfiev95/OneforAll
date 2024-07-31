import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def test(model, test_dataloader, criterion, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.load_state_dict(torch.load(opt.weight))
    except:
        params = torch.load(opt.weight)
        model.load_state_dict(params["weight"])
    val_loss_best = 0.0
    val_loss_avg = 0.0