from collections import namedtuple

# SUNDAY at 5 :D

# NOTES:
# - Do your implementations in NEW files (we'll compile them later)
# - Pass around strings
# - Use the class' Neural Network (we'll modify it later)

# TASKS:
# Task 1: Generate game instances and classify them -- Naomi
# Task 2: Implement Speaker module (produce() and learn()) -- Phil
# Task 3: Implement Listerner module (interpret() and learn())  -- Ciyang

# PROJECT MILESTONE:
# (due May 25, 11:59 pm)
# A short overview of your project including at least the following
# information:
#   - A statement of the project's goals.
#   - A summary of previous approaches (drawing on the lit review).
#   - A summary of the current approach.
#   - A summary of progress so far: what you have been done, what you still
#     need to do, and any obstacles or concerns that might prevent your project
#     from coming to fruition.

Game = namedtuple('Game', ['messages', 'targets', 'sems'])  # G = (M, T, [.])

Features = ['hats', 'glasses', 'mustache']

Messages = {
    # message: vector representation,
    'hats': [1, 0, 0],
    'glasses': [0, 1, 0],
    'mustache': [0, 0, 1],
    }

Targets = {
    # target: vector representation,
    'left': [1, 0, 0],
    'center': [0, 1, 1],  # targets can have multiple features
    'right': [0, 0, 1],
    }


class Listener(object):

    def __init__(self, games=None):
        self.games = games

    def interpret(game, message):
        # produces a target (from Targets dict)
        pass

    def learn():
        pass


class Speaker(object):

    def __init__(self, games=None):
        self.games = games

    def produce(game, target):
        # produces a message (from Messages dict)
        pass

    def learn():
        pass


def play(game):
    pass
