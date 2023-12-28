import board
import stack
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
    def __init__(self, board : board.Board, color : stack.Color, m=None):
        self.board = board
        self.color = color
        if m is None:
            self.model = model.Model()
        else:
            self.model = m
        self.path = []

    
    def turn(self, possible_moves=None):
        """
        take a turn for this player
        each turn MUST:
            EITHER call put_piece on the stack object they want to play on OR flip a stack
            remove the move from self.board.available_spots if they put the last piece on it
        """

        possible_moves = self.board.find_possible_moves()
        vs = [(m, self.board.test_move(self, m)) for m in possible_moves]

        current_node = tree.Node(
            board_vec=self.board.vector_repr(self),
            children_vectors= vs,
            model=self.model,
        )
        selected_node = current_node.select_child()

        self.path.append(selected_node)
        return selected_node.move_taken


    def give_reward(self, r):
        num_turns = len(self.path)
        reward_proportions = linspace(0.08, 1, num_turns)
        for i in range(len(self.path)-1, -1, -1):
            p = reward_proportions[i]
            # print(f"applying {p} proportion of reward {r} to the {i}th node in path")
            self.path[i].model.apply_reward(
                node_chosen=self.path[i],
                y_actual=r * p
            )
        self.reset()


    def reset(self):
        self.path = []
        # self.model = self.model.reset_model()