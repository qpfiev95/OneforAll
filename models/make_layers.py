import torch.nn as nn
import torch
from collections import  OrderedDict

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))

        elif 'linear' in layer_name:
            Linear = nn.Linear(in_features=v[0], out_features=v[1])
            layers.append((layer_name, Linear))
            if 'bn1d' in layer_name:
                batch_norm = nn.BatchNorm1d(num_features=v[0])
                layers.append(('bn_' + layer_name, batch_norm))
            elif 'in1d' in layer_name:
                instance_norm = nn.InstanceNorm1d(num_features=v[0])
                layers.append(('in_' + layer_name, instance_norm))

            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            elif 'gelu' in layer_name:
                layers.append(('gelu_' + layer_name, nn.GELU()))
            elif 'prelu' in layer_name:
                layers.append(('prelu_' + layer_name, nn.PReLU()))
            elif 'tanh' in layer_name:
                layers.append(('tanh_' + layer_name, nn.Tanh()))

        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'bn' in layer_name:
                batch_norm = nn.BatchNorm2d(num_features=v[1])
                layers.append(('bn_' + layer_name, batch_norm))
            elif 'in' in layer_name:
                instance_norm = nn.InstanceNorm2d(num_features=v[1])
                layers.append(('in_' + layer_name, instance_norm))

            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            elif 'gelu' in layer_name:
                layers.append(('gelu_' + layer_name, nn.GELU()))
            elif 'prelu' in layer_name:
                layers.append(('prelu_' + layer_name, nn.PReLU()))
            elif 'tanh' in layer_name:
                layers.append(('tanh_' + layer_name, nn.Tanh()))

        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'bn' in layer_name:
                batch_norm = nn.BatchNorm2d(num_features=v[1])
                layers.append(('bn_' + layer_name, batch_norm))
            elif 'in' in layer_name:
                instance_norm = nn.InstanceNorm2d(num_features=v[1])
                layers.append(('in_' + layer_name, instance_norm))

            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            elif 'gelu' in layer_name:
                layers.append(('gelu_' + layer_name, nn.GELU()))
            elif 'prelu' in layer_name:
                layers.append(('prelu_' + layer_name, nn.PReLU()))
            elif 'tanh' in layer_name:
                layers.append(('tanh_' + layer_name, nn.Tanh()))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))