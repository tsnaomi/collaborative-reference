from ibr_classifer import ibr_classifier

from collections import namedtuple
from itertools import combinations_with_replacement as cwr, permutations
from pprint import pprint

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

RefInstance = namedtuple('RefInstance', ['game', 'message', 'target'])

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

Sems = {
    'hats': ['left', ],
    'glasses': ['center', ],
    'mustache': ['center', 'right', ],
    }


def all_games():
    '''Produce a list of every possible Game instance.'''

    def get_targets(vector):
        return {'left': vector[:3], 'center': vector[3:6], 'right': vector[6:]}

    def get_sems(targets):
        sems = {0: [], 1: [], 2: [], }  # hats, glasses, mustache

        for k, v in targets.iteritems():
            for i, f in enumerate(v):
                if f:
                    sems[i].append(k)

        return {'hats': sems[0], 'glasses': sems[1], 'mustache': sems[2]}

    vectors = set([p for c in cwr(range(2), 9) for p in permutations(c)])
    vectors = [list(v) for v in vectors]
    targets = [get_targets(vector) for vector in vectors]
    return [Game(Messages, t, sems=get_sems(t)) for t in targets]


def all_reference_instances(games=all_games()):
    '''Produce a list of every possible reference instance given a set of games.

    A reference instance is a Game G, a message m, and intended target t.
    '''

    def get_ref(game, message, target):
        return {'game': game, 'message': message, 'target': target}

    messages = Messages.keys()
    targets = Targets.keys()

    return [get_ref(g, m, t) for g in games for m in messages for t in targets]


def classify_reference_instances(ref_instances=all_reference_instances()):
    '''Return three lists for level 0, 1, and 2 reference instances.'''
    level = {-1: [], 0: [], 1: [], 2: []}

    for i in ref_instances:
        level[ibr_classifier(**i)].append(i)

    return level[0], level[1], level[2]

level_0, level_1, level_2 = classify_reference_instances()


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
