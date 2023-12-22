import board
import stack
import random

class Player():
    def __init__(self, board : board.Board, color : stack.Color):
        self.board = board
        self.color = color
    
    def turn(self, possible_moves=None):
        """
        take a turn for this player
        each turn MUST:
            EITHER call put_piece on the stack object they want to play on OR flip a stack
            remove the move from self.board.available_spots if they put the last piece on it
        """
        if possible_moves is None:
            possible_moves = self.board.find_possible_moves()
            self.board.possible_moves = possible_moves
        
        move = random.choice(possible_moves)
        
        if move[0] == 0: # "place" move
            self.board.grid[move[1]][move[2]].put_piece(self.color)
            if self.board.grid[move[1]][move[2]].full():
                self.board.available_spots.remove(move)
        elif move[0] == 1:
            self.board.grid[move[1]][move[2]].flip()
            #flipping does not change the available moves ever
        else:
            raise ValueError(f"invalid move ID: '{move[0]}', must be 0 or 1")
    
    def reset(self):
        pass # shouldn't reset color, and board is reset seperately