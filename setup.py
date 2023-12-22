import board
import stack
import players
import random

class FourPlayerGame():
    def __init__(self):
        self.board = board.Board()
        self.color_1 = stack.Color('r', '.')
        self.player_1 = players.Player(self.board, self.color_1)
        self.color_2 = stack.Color('b', '&')
        self.player_2 = players.Player(self.board, self.color_2)
        self.color_3 = stack.Color('o', '-')
        self.player_3 = players.Player(self.board, self.color_3)
        self.color_4 = stack.Color('y', '#')
        self.player_4 = players.Player(self.board, self.color_4)
        self.order = [self.player_1, self.player_2, self.player_3, self.player_4]
        random.shuffle(self.order)
    
    def reset(self):
        self.board.reset()
        # for p in self.order:
        #     p.reset() # player reset does nothing rn
        random.shuffle(self.order)

    
    def play(self):
        for p in self.order:
            n = p.turn()

    