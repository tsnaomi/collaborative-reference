from collections import namedtuple
from ibr_classifier import ibr_classifier
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
    'left': [0, 0, 0],
    'center': [0, 0, 1],  # targets can have multiple features
    'right': [0, 0, 1],
    }

Sems = {
    'hats': [],
    'glasses': [],
    'mustache': ['center', 'right', ],
    }


def generate_classified_reference_instances(F=3, T=3):
    '''Generate and classify every possible reference instance.

    This function generates reference instances given F number of features and
    T number of targets. It then classifies these instances into three lists
    corresponding to Level 0, Level 1, and Level 2 instances.

    Classification is done using the ibr_classifer.
    '''
    messages = Messages if F == 3 else _create_messages_dict(F)
    games = all_games(messages, F, T)
    reference_instances = _all_reference_instances(games, messages, F, T)

    return _classify_reference_instances(reference_instances)


# this function produces a list of every possible Game instance, given F number
# of features and T number of targets
def all_games(messages=Messages, F=3, T=3):
    '''Produce a list of every possible Game instance.'''

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

        if F != 3 or T != 3:
            return {'m%s' % i: sems[i] for i in range(F)}

        return {'hats': sems[0], 'glasses': sems[1], 'mustache': sems[2]}

    vectors = [p for c in combin(range(2), F * T) for p in permutations(c)]
    targets = [get_targets(list(vector)) for vector in set(vectors)]

    return [Game(messages, t, get_sems(t)) for t in targets]


# this function produces a list of every possible reference instance given a
# list of games, F number of features, and T number of targets; each reference
# instance consists of a game, a message, and an intended target, returned in
# the form of a dictionary
def _all_reference_instances(games, messages, F, T):

    def get_ref(game, message, target):
        return {'game': game, 'message': message, 'target': target}

    messages = messages.keys()
    targets = Targets.keys() if T == 3 else ['t%s' % i for i in range(T)]

    return [get_ref(g, m, t) for g in games for m in messages for t in targets]


# this function creates a messages dictionary given F number of identifying
# features; e,g,, if F=4, this will return the following key-value pairs:
#       'm0': [1, 0, 0, 0],
#       'm1': [0, 1, 0, 0],
#       'm2': [0, 0, 1, 0],
#       'm3': [0, 0, 0, 1],
def _create_messages_dict(F):
    messages = {'m%s' % i: list(repeat(0, F)) for i in range(F)}

    for i in range(F):
        messages['m%s' % i][i] = 1

    return messages


# given a list of reference instances, this function returns three lists
# corresponding to Level 0, Level 1, and Level 2 instances, and discards any
# level -1 instances (see ibr_classifier)
def _classify_reference_instances(reference_instances):
    level = {-1: [], 0: [], 1: [], 2: []}

    for i in reference_instances:
        level[ibr_classifier(**i)].append(i)

    return level[-1], level[0], level[1], level[2]

unsolvable, level0, level1, level2 = generate_classified_reference_instances()


def Separable(game):
    # check whether a game is seperable
    # a game is separable iff there exists a bijection between messages and
    # targets such that each message-target pair is solvable

    messages = game.messages.keys()
    targets = game.targets.keys()

    for targets_in_some_order in permutations(targets):
        for message, target in zip(messages, targets_in_some_order):
            if ibr_classifier(game, message, target) == -1:
                break
        else:
            return True

    return False

# separable games used to self-train
sep_games = [game for game in all_games() if Separable(game)]
