import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Model():
    def __init__(self, lr=0.02):
        self.lr = lr
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.reset_model()


    def reset_model(self):

        self.model = nn.Sequential(
            # nn.Flatten(),
            nn.Conv2d(1, 8, kernel_size=[1,3], padding=(0, 0)),
            nn.Conv2d(8, 50, kernel_size=[2,2], padding=(1, 1)),
            nn.Conv2d(50, 1, kernel_size=[3,3], padding=(2, 2)),
            # nn.Conv2d(100, 140, kernel_size=[4,4]),
            nn.Flatten(),
            nn.Linear(91, 100),
            # nn.ReLU(),
            nn.Sigmoid(),
            # nn.LogSigmoid(),
            nn.Linear(100, 120),
            # nn.ReLU(),
            nn.Linear(120, 50),
            nn.Linear(50, 1),
            # nn.Sigmoid(),
            # nn.Softmax(),
        ).to(self.device)
        
        self.loss_fn = nn.BCELoss()  # binary cross entropy
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


    def apply_reward(self, node_chosen, y_actual):
        y_pred = self.calculate(torch.FloatTensor([node_chosen.board_vec]))
        y_actual = torch.FloatTensor([[y_actual]])
        self.loss = self.loss_fn(y_pred, y_actual)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def calculate(self, input):
        return self.model(input)