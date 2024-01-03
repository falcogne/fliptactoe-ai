import stack

MAX_NUM_FOR_STACK = 4
NUM_TO_WIN = 3

EMPTY_COLOR = " "
DEFAULT_SIZE = (4,4)

EXPLORATION_RATE = 0.3
LEARNING_RATE = 0.02


REWARD_WINNER = 1
REWARD_AT_FAULT = -1
REWARD_ACROSS = -0.01
REWARD_FARTHEST = 0.03


LOWEST_REWARD_PROPORTION = 0.02 # how much reward does the first move get
HIGHEST_REWARD_PROPORTION = 0.2 # how much reward does the second to last move get


hours = 0
minutes = 0
seconds = 40
TIME_TO_TRAIN = hours * 60 * 60 + minutes * 60 + seconds
WANT_TO_TRAIN = False