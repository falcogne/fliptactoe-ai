import constants

class Stack():
    def __init__(self):
        self.top_color = None
        self.bottom_color = None
        self.num_pieces = 0
    
    def flip(self):
        """flip this stack over"""
        self.top_color, self.bottom_color = self.bottom_color, self.top_color
    
    def put_piece(self, color):
        """add a piece to the top of this stack"""
        assert isinstance(color, Color)
        if self.full():
            raise ValueError("stack is already full, cannot add another piece")
        self.num_pieces += 1
        self.top_color = color
        if self.num_pieces == 1:
            self.bottom_color = color
    
    def full(self):
        return self.num_pieces == constants.MAX_NUM_FOR_STACK

class Color():
    def __init__(self, id, c):
        self.id = id
        self.printed = c
        assert len(self.id) == 1
        assert len(self.printed) == 1
    
    def __str__(self):
        return self.printed
    
    def __repr__(self):
        return str(self)
