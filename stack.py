import constants

class Stack():
    def __init__(self, tc=None, bc=None, np=0):
        if tc is None:
            self.top_color = Color(constants.EMPTY_COLOR, constants.EMPTY_COLOR)
        else:
            self.top_color = tc
        
        if bc is None:
            self.bottom_color = Color(constants.EMPTY_COLOR, constants.EMPTY_COLOR)
        else:
            self.bottom_color = bc
        
        self.num_pieces = np
    
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
        if self.bottom_color == constants.EMPTY_COLOR:
            self.bottom_color = color
    
    def full(self):
        return self.num_pieces == constants.MAX_NUM_FOR_STACK
    
    def copy(self):
        return Stack(self.top_color, self.bottom_color, self.num_pieces)

class Color():
    def __init__(self, id, c):
        self.id = id
        self.printed = c
        assert len(self.id) == 1
        assert len(self.printed) == 1
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Color):
            return self.id == other.id and self.printed == other.printed # TODO: maybe not the printed stuffs
        elif isinstance(other, str):
            return self.id == other

    

    def __str__(self):
        return self.printed
    

    def __repr__(self):
        return str(self)
