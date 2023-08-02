import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = [nn.Linear(input_size, hidden_size[0])]
        for i in range(len(hidden_size)):
            # skip the first layer
            if i == 0:
                continue
            if i == len(hidden_size) - 1:
                self.layers.append(nn.Linear(hidden_size[i], output_size))
            else:
                self.layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))


    def forward(self, x):
        x = F.relu(self.layers[0](x))
        for i in range(len(self.layers)):
            if i == 0:
                continue
            x = F.relu(self.layers[i](x))
        return x

        # x = F.relu(self.linear1(x))
        # x = self.linear2(x)
        # return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)



