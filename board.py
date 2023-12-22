import stack
import constants

class Board():
    def __init__(self, size=(4,4)):
        self.size = size
        self.reset()
    
    def reset(self):
        self.grid = [[stack.Stack() for _ in range(self.size[1])] for _ in range(self.size[0])]
        self.possible_moves = [(0, row, col) for row, arr in enumerate(self.grid) for col, _ in enumerate(arr)]
    
    def winner(self):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for row, arr in enumerate(self.grid):
            for col, stk in enumerate(arr):
                if stk.top_color == constants.EMPTY_COLOR:
                    continue
                # start at a point and check if any direction is the same color
                for d in directions:
                    # catch index errors
                    try:
                        num_in_row = 1
                        piece = (row, col)
                        while num_in_row < constants.NUM_TO_WIN and self.grid[piece[0]+d[0]][piece[1]+d[1]].top_color == stk.top_color:
                            num_in_row += 1
                            piece = (piece[0]+d[0], piece[1]+d[1])
                            if piece[0] < 0 or piece[1] < 0:
                                raise IndexError("you shouldn't be seeing this, it shoudl be caught in the except")
                        if num_in_row >= constants.NUM_TO_WIN:
                            return stk.top_color
                    except IndexError:
                        pass
        return None
    
    def find_possible_moves(self):
        moves = []
        for row, l in enumerate(self.grid):
            for col, stk in enumerate(l):
                if not stk.full():
                    moves.append((0, row, col))
                if stk.num_pieces > 0:
                    moves.append((1, row, col))
        return moves
    
    def __str__(self):
        """
        get code to do this but fill in t and b with the corresponding top and bottom colors:
        
        print("---------------------------------")
        print("|   t   |   t   |   t   |   t   |")
        print("|  (b)  |  (b)  |  (b)  |  (b)  |")
        print("---------------------------------")
        print("|   t   |   t   |   t   |   t   |")
        print("|  (b)  |  (b)  |  (b)  |  (b)  |")
        print("---------------------------------")
        print("|   t   |   t   |   t   |   t   |")
        print("|  (b)  |  (b)  |  (b)  |  (b)  |")
        print("---------------------------------")
        print("|   t   |   t   |   t   |   t   |")
        print("|  (b)  |  (b)  |  (b)  |  (b)  |")
        print("---------------------------------")
        """
        s = "---------------------------------"
        for arr in self.grid:
            s += "\n|   "
            for stk in arr:
                s+=f"{stk.top_color}   |   "
            s+="\n|  ("
            for stk in arr:
                s+=f"{stk.bottom_color})  |  ("
            s = s[:-3]
            s+="\n---------------------------------"
        return s
