import stack
import constants
import players
import model
from random import shuffle 
from torch import load as torchload
import time
import shutil

class Board():
    def __init__(self, size=constants.DEFAULT_SIZE, model_n=constants.STARTER_MODEL_NUM):
        self.size = size
        self.curr_model_n = model_n
        self.prev_model_n = model_n-1
        self.curr_model = model.Model(load=constants.LOAD_MODEL, square_fn=f'models/sq_{self.curr_model_n}.pt', threes_fn=f'models/th_{self.curr_model_n}.pt', linear_fn=f'models/ln_{self.curr_model_n}.pt')
        self.prev_model = model.Model(load=constants.LOAD_MODEL, square_fn=f'models/sq_{self.prev_model_n}.pt', threes_fn=f'models/th_{self.prev_model_n}.pt', linear_fn=f'models/ln_{self.prev_model_n}.pt')
        self.prev_model.learn = False

        self.color_1 = stack.Color('r', '.')
        self.color_2 = stack.Color('b', '&')
        self.color_3 = stack.Color('o', '-')
        self.color_4 = stack.Color('y', '#')

        # self.player_1 = players.Player(self, self.color_1)
        # self.player_2 = players.Player(self, self.color_2)
        # self.player_3 = players.Player(self, self.color_3)
        # self.player_4 = players.Player(self, self.color_4)
        
        self.exploration_rate = constants.EXPLORATION_RATE

        self.player_1 = players.VISIONPLAYER(self, self.color_1, self.curr_model, exploration_rate=self.exploration_rate)
        self.player_2 = players.VISIONPLAYER(self, self.color_2, self.curr_model, exploration_rate=self.exploration_rate)
        # self.player_3 = players.VISIONPLAYER(self, self.color_3, self.curr_model, exploration_rate=self.exploration_rate)
        # self.player_4 = players.VISIONPLAYER(self, self.color_4, self.curr_model, exploration_rate=self.exploration_rate)
        self.player_3 = players.VISIONPLAYER(self, self.color_3, self.prev_model, exploration_rate=0)
        self.player_4 = players.VISIONPLAYER(self, self.color_4, self.prev_model, exploration_rate=0)

        # self.curr_model_players = [self.player_1, self.player_2, self.player_3, self.player_4]
        self.curr_model_players = [self.player_1, self.player_2]
        self.prev_model_players = [self.player_3, self.player_4]
        self.curr_model_win_count = 0
        self.prev_model_win_count = 0
        self.same_model_in_a_row = 0

        self.order = [self.player_1, self.player_2, self.player_3, self.player_4]

        self.reset()


    def reset(self):
        self.grid = [[stack.Stack() for _ in range(self.size[1])] for _ in range(self.size[0])]
        self.possible_moves = [(0, row, col) for row, arr in enumerate(self.grid) for col, _ in enumerate(arr)]
        
        shuffle(self.order)
        self.turn_i = 0

        for p in self.order:
            p.reset()

    def swap_models(self):
        self.curr_model.save()
        self.prev_model.save()

        print(f"current [dynamic {self.curr_model_n}] win count: {self.curr_model_win_count} | prev [static {self.prev_model_n}]: {self.prev_model_win_count}")
        if self.curr_model_win_count > self.prev_model_win_count:
            self.same_model_in_a_row = 0
            self.prev_model_n = self.curr_model_n
            self.curr_model_n += 1
            self.prev_model.learn = False

            # Copy the current model to the new slot (initialize new iteration)
            shutil.copy(f'models/sq_{self.prev_model_n}.pt', f'models/sq_{self.curr_model_n}.pt')
            shutil.copy(f'models/th_{self.prev_model_n}.pt', f'models/th_{self.curr_model_n}.pt')
            shutil.copy(f'models/ln_{self.prev_model_n}.pt', f'models/ln_{self.curr_model_n}.pt')

            print(f"starting new model [{self.curr_model_n}] from copy of [{self.prev_model_n}]")
        else:
            self.same_model_in_a_row += 1
            self.curr_model_n = self.prev_model_n
            print(f"new was worse, so going back to [{self.curr_model_n}] and trying learning again")
        
        self.curr_model_win_count = 0
        self.prev_model_win_count = 0
        
        self.curr_model = model.Model(load=True, square_fn=f'models/sq_{self.curr_model_n}.pt', threes_fn=f'models/th_{self.curr_model_n}.pt', linear_fn=f'models/ln_{self.curr_model_n}.pt')
        self.prev_model = model.Model(load=True, square_fn=f'models/sq_{self.prev_model_n}.pt', threes_fn=f'models/th_{self.prev_model_n}.pt', linear_fn=f'models/ln_{self.prev_model_n}.pt')
        self.prev_model.learn=False

            
        if self.same_model_in_a_row > 4:
            self.exploration_rate *= 0.9
            print(f"lowering exploration rate to {self.exploration_rate:.4}")
            self.same_model_in_a_row = 0
            
        for p in self.curr_model_players:
            p.model = self.curr_model
            p.exploration_rate = self.exploration_rate
        for p in self.prev_model_players:
            p.model = self.prev_model

    def test_move(self, player, move):
        stk = self.grid[move[1]][move[2]]
        orig_stack = stk.copy()
    
        # do the move on the stack (but don't update availables because we're gonna reset it back)
        if move[0] == 0: # "place" move
            stk.put_piece(player.color)
        elif move[0] == 1:
            stk.flip()
        else:
            raise ValueError(f"invalid move ID: '{move[0]}', must be 0 or 1")

        # get the updated test version of the board
        v = self.vector_repr(player)

        # reset board with original stack
        self.grid[move[1]][move[2]] = orig_stack

        return v


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
        if len(self.possible_moves) != 0:
            return self.possible_moves
        
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
        s = "--------" * self.size[1] + "-"
        for arr in self.grid:
            s += "\n| "
            for stk in arr:
                s+=f"{stk.num_pieces if stk.num_pieces != 0 else ' '} {stk.top_color}   | "
            s+="\n|  ("
            for stk in arr:
                s+=f"{stk.bottom_color})  |  ("
            s = s[:-3]
            s += "\n" + "--------" * self.size[1] + "-"
        return s

    def vector_repr(self, p) -> list[int]:
        m = []
        my_i = self.order.index(p)
        

        me = self.order[my_i]
        next_player = self.order[(my_i+1) % len(self.order)]
        across_player = self.order[(my_i+2) % len(self.order)]
        before_player = self.order[(my_i+3) % len(self.order)]
        for arr in self.grid:
            vec = []
            for stk in arr:
                if me.color == stk.top_color:
                    # vec.append(1)
                    vec.extend([1,0,0,0])
                elif next_player.color == stk.top_color:
                    # vec.append(2)
                    vec.extend([0,1,0,0])
                elif across_player.color == stk.top_color:
                    # vec.append(3)
                    vec.extend([0,0,1,0])
                elif before_player.color == stk.top_color:
                    # vec.append(4)
                    vec.extend([0,0,0,1])
                else:
                    # vec.append(0)
                    vec.extend([0,0,0,0])

                if me.color == stk.bottom_color:
                    # vec.append(1)
                    vec.extend([1,0,0,0])
                elif next_player.color == stk.bottom_color:
                    # vec.append(2)
                    vec.extend([0,1,0,0])
                elif across_player.color == stk.bottom_color:
                    # vec.append(3)
                    vec.extend([0,0,1,0])
                elif before_player.color == stk.bottom_color:
                    # vec.append(4)
                    vec.extend([0,0,0,1])
                else:
                    # vec.append(0)
                    vec.extend([0,0,0,0])

                one_hot = [0,0,0,0]
                one_hot[stk.num_pieces-1] = 1
                vec.extend(one_hot)

            m.append(vec)
        return m


    def apply_move(self, player, move):
        if move[0] == 0: # "place" move
            self.grid[move[1]][move[2]].put_piece(player.color)
            if self.grid[move[1]][move[2]].full():
                self.possible_moves.remove(move)
            if self.grid[move[1]][move[2]].num_pieces == 1:
                self.possible_moves.append((1, move[1], move[2]))
        elif move[0] == 1:
            self.grid[move[1]][move[2]].flip()
            #flipping does not change the available moves ever
        else:
            raise ValueError(f"invalid move ID: '{move[0]}', must be 0 or 1")


    def play(self, printed=False):
        if printed:
            print("-"*10 + "starting to play" + "-"*10)
        while self.winner() is None:
            p = self.order[self.turn_i]
            m = p.turn()
            self.apply_move(p, m)
            self.turn_i = (self.turn_i+1) % len(self.order)
            if printed:
                print(self)
                print(p.color, m, [self.order[(self.turn_i + i) % len(self.order)].color for i in range(3)])
                input()
        
        winning_color = self.winner()
        win_i = None
        winning_player = None
        for i, p in enumerate(self.order):
            if p.color == winning_color:
                win_i = i
                winning_player = p
                break
        if winning_player in self.curr_model_players:
            self.curr_model_win_count += 1
        elif winning_player in self.prev_model_players:
            self.prev_model_win_count += 1
        else:
            raise ValueError(f"winning player {winning_player} not listed for curr or prev model")
        at_fault = self.order[(win_i+3) % len(self.order)]
        across = self.order[(win_i+2) % len(self.order)]
        farthest = self.order[(win_i+1) % len(self.order)]

        bs, rs = winning_player.give_reward(constants.REWARD_WINNER)
        # BOARDS.extend(bs)
        # REWARDS.extend(rs)
        bs, rs = at_fault.give_reward(constants.REWARD_AT_FAULT)
        # BOARDS.extend(bs)
        # REWARDS.extend(rs)
        bs, rs = across.give_reward(constants.REWARD_ACROSS)
        # BOARDS.extend(bs)
        # REWARDS.extend(rs)
        bs, rs = farthest.give_reward(constants.REWARD_FARTHEST)
        # BOARDS.extend(bs)
        # REWARDS.extend(rs)

        if printed:
            # print(self.vector_repr(p))
            print(self)
            print("*"*10 + f"game over: winner is {self.winner()} " + "*"*10)
        
        return self.winner()

def hms(seconds):
    hrs = int(seconds / 60 / 60)
    min = int(seconds / 60 - (hrs * 60))
    sec = seconds - (hrs * 60 * 60) - (min * 60)
    return f"{hrs:02d}:{min:02d}:{sec:>06.3f}"


# import os
# import numpy as np
# from pympler.asizeof import asizeof
# BOARDS = []
# REWARDS = []
if __name__ == "__main__":
    start_time = time.time()
    b = Board()
    winners = {}
    end_print = "no training happened"

    if constants.WANT_TO_TRAIN:
        time_played = time.time() - start_time
        i = 0
        print(f"\ngoing to train for {hms(constants.TIME_TO_TRAIN)}\n")
        while time_played < constants.TIME_TO_TRAIN:
        # num_games_to_play = constants.NUMBER_TRAINING_GAMES
        # for i in range(num_games_to_play):
            if i != 0 and i % constants.SAVE_INTERVAL == 0:
                print(f"\nthrough {i} games in {hms(time_played)}")
                print(winners)
                b.swap_models()
                # b.curr_model.save()
            w = b.play()
            b.reset()

            try:
                winners[w.printed] += 1
            except KeyError:
                winners[w.printed] = 1

            time_played = time.time() - start_time
            i+=1
        else:
            b.curr_model.save()
            print("final save of model done")

        end_print = f"done training in {hms(time.time() - start_time)}, played {i} games"

    for i in range(constants.NUM_PRINT_AFTER_TRAINING):
        if i % 1000 == 0:
            print(f"{time.time():<23,} | starting game number {i}")
        w = b.play(printed=True)
        try:
            winners[w.printed] += 1
        except KeyError:
            winners[w.printed] = 1
        b.reset()
        # input()
    
    # np.savez('data.npz', arrays=BOARDS, rewards=REWARDS)
    # print("sizes")
    # print(f"{asizeof(BOARDS):,}")
    # print(f"{asizeof(REWARDS):,}")


    # loaded = np.load('data.npz', allow_pickle=True)
    # arrays = loaded['arrays']
    # rewards = loaded['rewards']

    # print(len(arrays))
    # print(arrays[:3])
    # print(asizeof(arrays))
    # print()
    # print(len(rewards))
    # print(rewards[:3])
    # print(asizeof(rewards))
    # raise NotImplementedError("done")

    print(winners)
    
    print(end_print)
