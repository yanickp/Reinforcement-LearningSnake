import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np



class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # Create a list to store the hidden layers
        self.hidden_layers = nn.ModuleList()

        # Add the first hidden layer
        if len(hidden_size) > 0:
            print("input size: ", input_size)
            self.input_layer = nn.Linear(input_size, hidden_size[0])
            print('hidden size 0: ', hidden_size[0])

        for idx, value in enumerate(hidden_size):
            if idx == 0:
                continue
            print('hidden size {idx}: ', hidden_size[idx])
            self.hidden_layers.append(nn.Linear(hidden_size[idx - 1], hidden_size[idx]))

        # Create the output layer
        print("output size: ", output_size)
        self.output_layer = nn.Linear(hidden_size[-1] if len(hidden_size) > 0 else input_size, output_size)

        # self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for i in range(len(self.hidden_layers)):
            x = F.relu(self.hidden_layers[i](x))
        x = self.output_layer(x) # todo activation function?
        return x

        # x = F.relu(self.linear1(x))
        # x = self.linear2(x)
        # return x

    def save(self, file_name='deepQmodel.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, targetNetwork=True):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = model
        self.targetNetwork = targetNetwork
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_step(self, state, action, reward, next_state, done):
        # state = torch.tensor(state, dtype=torch.float)
        state_combined = np.array(state)
        next_state_combined = np.array(next_state)
        # Convert the combined NumPy array to a PyTorch tensor
        state = torch.tensor(state_combined, dtype=torch.float)
        # next_state = torch.tensor(next_state, dtype=torch.float)
        next_state = torch.tensor(next_state_combined, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                if self.targetNetwork:
                    Q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx]))
                else:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
