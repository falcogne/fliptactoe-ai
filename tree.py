from random import choice, random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Node():
    def __init__(self, board_vec, children_vectors, model=None, exploration_rate=0.2, move_taken=None):
        self.board_vec = board_vec
        if model is None:
            self.reset_model()
        else:
            self.model = model
        self.N = 0
        self.Q = None
        self.exploration_rate = exploration_rate
        self.children = [Node(
            board_vec=v,
            children_vectors=[],
            model=self.model,
            move_taken=m
        ) for m, v in children_vectors]
        self.model_output = None
        self.move_taken = move_taken
        self.child_selected = None
    
    def select_child(self):
        for c in self.children:
            t = torch.FloatTensor([c.board_vec]) # 1 channel input
            # print(t)
            c.model_output = self.model.calculate(t)
            # print(c.model_output)

        # explore random option
        if random() < self.exploration_rate:
            self.child_selected = choice(self.children)
        else:
            self.child_selected = max(self.children, key=lambda x:x.model_output[0])

        return self.child_selected