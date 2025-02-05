import board
import stack
import constants

import random
import tree
import model
from numpy import linspace


"""
MODEL:
have a model to evaluate:
    Having just made a move, how good is this board for me winnng
    this model is only gonna be trained on boards where the current player has just taken their move
        should output that player's chance of winning based on that board
    
PLAYER:
each player has a list for their "path" in the game: the board state after each move they've played
        NODE: each path will be made of nodes: board state (before move), model, children options, exploration rate
   3
 4   2
   1

1 is you;
    if 1 wins, reward 1,
    2 : -1
    3 : 0
    4 : 0


WITHIN PLAYER LIST CALL FUNCTION ON EACH NODE
Once someone wins, backpropogare the appropriate reward to each player's path
    pass a value due to math to each element in the path and use that as the "actual" reward to the model's output
    TODO: this math is gonna be hard


playing a game:

whle game is still going:
    create node for game rn
    take your turn
    store node in path

once game is done:
    pass reward to each player
    within each player, go through path backwards and pass appropriate reward

"""


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
        
        return move
    

    def give_reward(self, r):
        pass


    def reset(self):
        pass # shouldn't reset color, and board is reset seperately

class VISIONPLAYER(Player):
    def __init__(self, board : board.Board, color : stack.Color, m=None, exploration_rate=constants.EXPLORATION_RATE):
        self.board = board
        self.color = color
        if m is None:
            self.model = model.Model()
        else:
            self.model = m
        self.path = []
        self.exploration_rate = exploration_rate

    
    def turn(self, possible_moves=None):
        """
        take a turn for this player
        each turn MUST:
            EITHER call put_piece on the stack object they want to play on OR flip a stack
            remove the move from self.board.available_spots if they put the last piece on it
        """

        possible_moves = self.board.find_possible_moves()
        vs = [(m, self.board.test_move(self, m)) for m in possible_moves]
        # i = random.randint(0, len(possible_moves)-1)
        # self.path.append(vs[i][1])
        # return vs[i][0]

        current_node = tree.Node(
            board_vec=self.board.vector_repr(self),
            children_vectors= vs,
            model=self.model,
            exploration_rate=self.exploration_rate
        )
        selected_node = current_node.select_child()

        self.path.append(selected_node)
        return selected_node.move_taken


    def give_reward(self, r):
        num_turns = len(self.path)
        reward_proportions = linspace(constants.LOWEST_REWARD_PROPORTION, constants.HIGHEST_REWARD_PROPORTION, num_turns-1)
        reward = [r*p for p in reward_proportions] + [r]
        # reward = [r for _ in reward_proportions] + [r]
        # print(reward)
        bs = []
        rs = []
        for i in range(len(self.path)-1, -1, -1):
            # print(f"applying {p} proportion of reward {r} to the {i}th node in path")
            bs.append(self.path[i])
            rs.append(reward[i])
            self.path[i].model.apply_reward(
                node_chosen=self.path[i],
                y_actual=reward[i]
            )
        self.reset()
        return bs, rs


    def reset(self):
        self.path = []
        # self.model = self.model.reset_model()