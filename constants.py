import stack

MAX_NUM_FOR_STACK = 4
NUM_TO_WIN = 3

EMPTY_COLOR = " "
DEFAULT_SIZE = (4,4)

EXPLORATION_RATE = 0.2
LEARNING_RATE = 0.001

REWARD_WINNER = 1
REWARD_AT_FAULT = -1
REWARD_ACROSS = -0.09
REWARD_FARTHEST = -0.02
# REWARD_ACROSS = -0.5
# REWARD_FARTHEST = -0.3


LOWEST_REWARD_PROPORTION = 0.02 # how much reward does the first move get
HIGHEST_REWARD_PROPORTION = 0.8 # how much reward does the second to last move get

SAVE_INTERVAL = 200 # games before it prints/saves

hours = 0
minutes = 10
seconds = 0
TIME_TO_TRAIN = hours * 60 * 60 + minutes * 60 + seconds
WANT_TO_TRAIN = True
LOAD_MODEL = True
STARTER_MODEL_NUM = 9  # curr model num so >= 1
NUM_PRINT_AFTER_TRAINING = 1

# save models and play them against each other
# save ___ and feed those values into model
# - "how many in a row"
#   - for you and the next person
# - "top and bottom color of stack"???? prolly not
# general advice: can't have convolution learn anything particular, just need to go deeper
# so do that ^ and have padding?