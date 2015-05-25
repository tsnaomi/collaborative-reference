from collections import namedtuple
from ibr_classifer import ibr_classifier
from itertools import (
    combinations_with_replacement as combin,
    permutations,
    repeat,
    )

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

Sems = {
    'hats': ['left', ],
    'glasses': ['center', ],
    'mustache': ['center', 'right', ],
    }


def generate_classified_reference_instances(F=3, T=3):
    '''Generate and classify every possible reference instance.

    This function generates reference instances given F number of features and
    T number of targets. It then classifies these instances into three lists
    corresponding to Level 0, Level 1, and Level 2 instances

    Classification is done using the ibr_classifer.
    '''
    games = _all_games(F, T)
    reference_instances = _all_reference_instances(games, F, T)

    return _classify_reference_instances(reference_instances)


def _all_games(F, T):
    # this function produces a list of every possible Game instance, given F
    # number of features and T number of targets

    def get_targets(vector):
        if F != 3 or T != 3:
            return {'t%s' % i: vector[i * F:(i * F) + F] for i in range(t)}

        return {'left': vector[:3], 'center': vector[3:6], 'right': vector[6:]}

    def get_sems(targets):
        sems = {i: [] for i in range(F)}

        for k, v in targets.iteritems():
            for i, f in enumerate(v):
                if f:
                    sems[i].append(k)

        if f != 3 or t != 3:
            return {'m%s' % i: sems[i] for i in range(f)}

        return {'hats': sems[0], 'glasses': sems[1], 'mustache': sems[2]}

    vectors = [p for c in combin(range(2), F * T) for p in permutations(c)]
    vectors = [list(v) for v in set(vectors)]
    targets = [get_targets(vector) for vector in vectors]

    return [Game(Messages, t, sems=get_sems(t)) for t in targets]


def _all_reference_instances(games, F, T):
    # this function produces a list of every possible reference instance given
    # a list of games, F number of features, and T number of targets

    # each reference instance is a game 'game', a message 'message', and an
    # an intended target 'target', returned in the form of a dictionary

    def get_ref(game, message, target):
        return {'game': game, 'message': message, 'target': target}

    messages = _create_messages_dict(F=F).keys()
    targets = Targets.keys() if T == 3 else ['t%s' % i for i in range(T)]

    return [get_ref(g, m, t) for g in games for m in messages for t in targets]


def _create_messages_dict(F):
    # this function creates a messages dictionary given F number of identifying
    # features

    # e,g,, if F=4, this will return the following key-value pairs:
    #       'm0': [1, 0, 0, 0],
    #       'm1': [0, 1, 0, 0],
    #       'm2': [0, 0, 1, 0],
    #       'm3': [0, 0, 0, 1],

    if F == 3:
        return Messages

    messages = {'m%s' % i: list(repeat(0, F)) for i in range(F)}

    for i in range(F):
        messages['m%s' % i][i] = 1

    return messages


def _classify_reference_instances(ref_instances):
    '''Return three lists of level 0, 1, and 2 reference instances.'''
    # given a list of reference instances, this function returns three lists
    # corresponding to Level 0, Level 1, and Level 2 instances, and discards
    # level -1 instances (see ibr_classifier)

    level = {-1: [], 0: [], 1: [], 2: []}

    for i in ref_instances:
        level[ibr_classifier(**i)].append(i)

    return level[0], level[1], level[2]

level_0, level_1, level_2 = generate_classified_reference_instances()


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
