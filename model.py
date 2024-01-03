import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import constants

class Model(nn.Module):
    def __init__(self, model_filename=None, lr=constants.LEARNING_RATE):
        super(Model, self).__init__()
        self.lr = lr
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        
        self.reset_model(model_filename)


    def reset_model(self, m_fn):
        if m_fn is not None:
            with open(m_fn, 'rb') as file:
                self.model = torch.load(file)
        else:
            self.model = nn.Sequential(
                # nn.Flatten(),
                nn.Conv2d(1, 8, kernel_size=[1,3], padding=(0, 0)),
                # nn.Conv2d(8, 50, kernel_size=[2,2], padding=(1, 1)),
                nn.Conv2d(8, 50, kernel_size=[3,3], padding=(0,0)),
                # nn.Conv2d(100, 1, kernel_size=[4,4]),
                nn.Flatten(start_dim=0),
                nn.Linear(800, 2000),
                nn.Tanh(),
                nn.Dropout(p=0.1),
                nn.Linear(2000, 600),
                nn.Tanh(),
                nn.Linear(600, 1),
                nn.Tanh(),
                # nn.ReLU(),
                # nn.LogSigmoid(),
                # nn.Softmax(),
            ).to(self.device)
        
        # self.loss_fn = nn.BCELoss()  # binary cross entropy
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


    def apply_reward(self, node_chosen, y_actual):
        y_pred = self.forward(torch.FloatTensor([node_chosen.board_vec]))
        y_actual = torch.FloatTensor([y_actual])
        self.loss = self.loss_fn(y_pred, y_actual)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def forward(self, input):
        return self.model(input)
    
    def save(self):
        with open("model.pt", 'wb') as m:
            torch.save(self, m)