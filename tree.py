from random import choice, random

import constants

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Node():
    def __init__(self, board_vec, children_vectors, model=None, exploration_rate=constants.EXPLORATION_RATE, move_taken=None):
        self.board_vec = board_vec
        if model is None:
            self.reset_model()
        else:
            self.model = model
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
            c.model_output = self.model.forward(t)
            # print(c.model_output)

        # explore random option
        if random() < self.exploration_rate:
            self.child_selected = choice(self.children)
        else:
            # print([c.model_output[0] for c in self.children])
            self.child_selected = max(self.children, key=lambda x:x.model_output[0])
            if (self.child_selected.model_output[0] == 1 and len([c for c in self.children if c.model_output[0] != 1]) == 0)\
               or\
               (self.child_selected.model_output[0] == -1 and len([c for c in self.children if c.model_output[0] != -1]) == 0):
                print(c.board_vec)
                print(c.move_taken)
                print([c.model_output for c in self.children])
                raise ValueError("It's become overtrained now I think, layer is all 1s")

        return self.child_selected