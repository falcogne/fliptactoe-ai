import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import constants

class Model(nn.Module):
    def __init__(self, load=False, lr=constants.LEARNING_RATE, square_fn='models/square.pt', threes_fn='models/threes.pt', linear_fn='models/linear.pt'):
        super(Model, self).__init__()
        self.lr = lr
        self.learn = True
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        
        self.load = load
        self.square_fn = square_fn
        self.threes_fn = threes_fn
        self.linear_fn = linear_fn
        self.reset_model()


    def reset_model(self):
        if self.load:
            self.conv2d_square = torch.load(self.square_fn)
            self.conv2d_threes = torch.load(self.threes_fn)
            self.model = torch.load(self.linear_fn)
            # with open(m_fn, 'rb') as file:
            #     self.model = torch.load(file)
                # self.load_state_dict(torch.load(file, map_location=self.device))
        else:
            print("init")
            self.conv2d_square = nn.Conv2d(1, 20, kernel_size=[1,12], padding=(0,0), stride=[1, 12])
            self.conv2d_threes = nn.Conv2d(1, 50, kernel_size=[3,36], padding=(0,0), stride=[1, 12])
            self.model = nn.Sequential(
                nn.Flatten(start_dim=0),
                nn.Linear(520, 1100),
                nn.Tanh(),
                nn.Linear(1100, 800),
                nn.Tanh(),
                nn.Linear(800, 1),
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
        if not self.learn:
            return
        y_pred = self.forward(torch.FloatTensor([node_chosen.board_vec]))
        y_actual = torch.FloatTensor([y_actual])
        self.loss = self.loss_fn(y_pred, y_actual)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


    def forward(self, x):
        x = x.unsqueeze(1)
        # print(np.shape(x))
        # print(x)
        # # Assume input x has shape (batch_size, 1, 4, 12)
        
        # Process input through the first convolutional path
        out1 = self.conv2d_square(x)
        out1 = out1.view(out1.size(0), -1)  # Flatten

        # Process input through the second convolutional path
        out2 = self.conv2d_threes(x)
        out2 = out2.view(out2.size(0), -1)  # Flatten
        
        # Combine the outputs from both paths
        combined_out = torch.cat((out1, out2), dim=1)  # Concatenate along the channel dimension
        combined_out = combined_out.view(combined_out.size(0), -1)  # Shape: [batch_size, 16*4*12]

        # Pass the combined output through the rest of the model
        out = self.model(combined_out)
        
        return out

    
    def save(self):
        # with open("model.pt", 'wb') as m:
        torch.save(self.conv2d_square, self.square_fn)
        torch.save(self.conv2d_threes, self.threes_fn)
        torch.save(self.model, self.linear_fn)
            # torch.save(self.state_dict(), m)