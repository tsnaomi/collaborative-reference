import os
import xml.etree.ElementTree as ET

from games import Game
from ibr_classifier import ibr_classifier
from itertools import combinations, product, repeat
from math import sqrt
from pdb import set_trace
from pprint import pprint


CORPUS = 'TUNA/corpus/singular/furniture'

# TODO: Implement object inheritance


# 14-dimensional vectors
class Tuna:

    def __init__(self, corpus=CORPUS):
        self.games = []
        self.reference_instances = []
        self.unsolvable = []
        self.level0 = []
        self.level1 = []
        self.level2 = []

        self.TunaFeatures = [
            'chair', 'sofa', 'desk', 'fan',  # type
            'blue', 'red', 'green', 'grey',  # colour
            'left', 'right', 'front', 'back',  # orientation
            'large', 'small',  # size
            ]

        self.T = 7  # each context contains 1 target and 6 distractors
        self.F = 14

        self.gather_reference_instances()
        self.classify()

    def key(self, x):
        # indices of type, colour, orientation, and size
        return self.TunaFeatures.index(x)

    def gather_reference_instances(self):
        for dirpath, dirname, filenames in os.walk(CORPUS):

            for f in filenames:
                filepath = dirpath + '/' + f
                ref_inst = self.create_reference_instance(filepath)

                if ref_inst:
                    self.reference_instances.append(ref_inst)

    def create_reference_instance(self, filepath):
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            # create targets
            target, referents = self._get_target_and_referents(root)

            # create message
            M, message = self._get_complex_message(root)

            # create messages
            messages = self._get_complex_messages_dict(M)

            if message not in messages.keys():
                raise KeyError('%s has ambiguous attributes.' % filepath)

            # create semantics
            semantics = self._get_semantics_dict(referents, messages)

            # create game
            game = Game(targets=referents, messages=messages, sems=semantics)

            # create reference instance
            ref_inst = {'game': game, 'message': message, 'target': target}

            # add game to list of all games
            self.games.append(game)

            return ref_inst

        # ignore non-xml files
        except ET.ParseError:
            pass

        # ignore xml files that have multiple values for a single attribute
        except KeyError as e:
            print e

    def _get_target_and_referents(self, root):
        # create referents
        referents = {}

        for entity in root.iter('ENTITY'):
            referent = entity.attrib['ID']
            vector = list(repeat(0, self.F))

            attributes = entity.getchildren()

            # set target
            if entity.attrib['TYPE'] == 'target':
                target = referent

            # colour, type, orientation, and size attributes
            for attr in attributes[:4]:
                vector[self.key(attr.attrib['VALUE'])] = 1

            referents[referent] = vector

        return target, referents

    def _get_complex_message(self, root):

        def non_dimension(a):
            return a.attrib['NAME'] not in 'x-dimension y-dimension'

        attr_set = root.find('ATTRIBUTE-SET').getchildren()
        attr_set = [a.attrib['VALUE'] for a in attr_set if non_dimension(a)]
        attr_set = filter(lambda a: a not in 'other unknown', attr_set)
        attr_set = [a for a in sorted(attr_set, key=self.key)]

        M = len(attr_set)

        message = ' '.join(attr_set)

        return M, message

    def _get_complex_messages_dict(self, M):

        def create_features_matrix(f):
            messages = [list(repeat(0, f)) for i in range(f + 1)]

            for i in range(f):
                messages[i][i] = 1

            return messages

        def flatten(li):
            return [i for feature in li for i in feature]

        def from_vector_to_message(v):
            message = [self.TunaFeatures[i] for i, w in enumerate(v) if w]

            return ' '.join(message)

        FeaturesMatrices = [
            create_features_matrix(4),  # type
            create_features_matrix(4),  # colour
            create_features_matrix(4),  # orientation
            create_features_matrix(2),  # size
            ]

        messages = [flatten(p) for p in product(*FeaturesMatrices)]
        messages = filter(lambda m: m.count(1) == M, messages)
        messages_dict = {from_vector_to_message(v): v for v in messages}

        return messages_dict

    def _get_semantics_dict(self, referents, messages):
        semantics = {m: [] for m in messages}

        for mk, mv in messages.iteritems():

            for tk, tv in referents.iteritems():
                if len(filter(None, map(lambda t, m: t * m, tv, mv))) == 3:
                    semantics[mk].append(tk)

        return semantics

    def classify(self):
        level = {-1: [], 0: [], 1: [], 2: []}

        for ref_inst in self.reference_instances:
            level[ibr_classifier(ref_inst, arg_max=True)].append(ref_inst)

        self.unsolvable.extend(level[-1])
        self.level0.extend(level[0])
        self.level1.extend(level[1])
        self.level2.extend(level[2])


# 16-dimensional vectors
class DimensionalTuna:

    def __init__(self, corpus=CORPUS):
        self.games = []
        self.reference_instances = []
        self.unsolvable = []
        self.level0 = []
        self.level1 = []
        self.level2 = []

        self.TunaFeatures = [
            'chair', 'sofa', 'desk', 'fan',  # type
            'blue', 'red', 'green', 'grey',  # colour
            'left', 'right', 'front', 'back',  # orientation
            'large', 'small',  # size
            'x-dimension',  # x-dimension
            'y-dimension',  # y-dimension
            ]

        self.T = 7  # each context contains 1 target and 6 distractors
        self.F = 16

        self.gather_reference_instances()
        self.classify()

    def key(self, x):
        try:
            # indices of type, colour, orientation, and size
            return self.TunaFeatures.index(x)

        except ValueError:
            # indices of x- and y- dimensions
            return 14 if int(x) in [0, 1, 2, 3, 4, 5] else 15

    def gather_reference_instances(self):
        for dirpath, dirname, filenames in os.walk(CORPUS):

            for f in filenames:
                filepath = dirpath + '/' + f
                ref_inst = self.create_reference_instance(filepath)

                if ref_inst:
                    self.reference_instances.append(ref_inst)

    def create_reference_instance(self, filepath):
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            # create targets
            target, referents = self._get_target_and_referents(root)

            # create message
            M, message = self._get_complex_message(root)

            # create messages
            messages = self._get_complex_messages_dict(M)

            if message not in messages.keys():
                raise KeyError('%s has ambiguous attributes.' % filepath)

            # create semantics
            semantics = self._get_semantics_dict(referents, messages)

            # create game
            game = Game(targets=referents, messages=messages, sems=semantics)

            # create reference instance
            ref_inst = {'game': game, 'message': message, 'target': target}

            # add game to list of all games
            self.games.append(game)

            return ref_inst

        # ignore non-xml files
        except ET.ParseError:
            pass

        # ignore xml files that have multiple values for a single attribute
        except KeyError as e:
            print e

    def _get_target_and_referents(self, root):
        # create referents
        referents = {}

        for entity in root.iter('ENTITY'):
            referent = entity.attrib['ID']
            vector = list(repeat(0, self.F))

            attributes = entity.getchildren()

            # set target
            if entity.attrib['TYPE'] == 'target':
                target = referent

            # colour, type, orientation, and size attributes
            for attr in attributes[:4]:
                vector[self.key(attr.attrib['VALUE'])] = 1

            # x-dimension attribute
            try:
                vector[14] = int(attributes[4].attrib['VALUE'])
            except ValueError:
                vector[14] = 0

            # y-dimension attribute
            try:
                vector[15] = int(attributes[5].attrib['VALUE']) + 5
                # y = y + 5 if y != 0 else -1
                # vector[15] = y
            except ValueError:
                vector[15] = -1

            referents[referent] = vector

        return target, referents

    def _get_complex_message(self, root):
        attr_set = []
        attributes = root.find('ATTRIBUTE-SET').getchildren()

        for a in attributes:

            try:
                num = int(a.attrib['VALUE'])

                if a.attrib['NAME'] == 'y-dimension':
                    num = -1 if num == 0 else num + 5

                attr_set.append(str(num))

            except ValueError:
                attr_set.append(a.attrib['VALUE'])

        attr_set = filter(lambda a: a not in 'other unknown', attr_set)
        attr_set = [a for a in sorted(attr_set, key=self.key)]

        M = len(attr_set)

        message = ' '.join(attr_set)

        return M, message

    def _get_complex_messages_dict(self, M):

        def create_features_matrix(f):
            messages = [list(repeat(0, f)) for i in range(f + 1)]

            for i in range(f):
                messages[i][i] = 1

            return messages

        def flatten(li):
            return [i for feature in li for i in feature]

        def is_viable(m):
            return m.count(0) + m.count(-1) == self.F - M,

        def from_vector_to_message(v):
            message = [self.TunaFeatures[i] for i, w in enumerate(v[:-2]) if w]
            message += [str(num) for num in v[-2:] if num > 0]

            return ' '.join(message)

        FeaturesMatrices = [
            create_features_matrix(4),  # type
            create_features_matrix(4),  # colour
            create_features_matrix(4),  # orientation
            create_features_matrix(2),  # size
            [[0, ], [1, ], [2, ], [3, ], [4, ], [5, ]],  # x_dimension
            [[-1, ], [6, ], [7, ], [8, ]],  # y-dimension
            ]

        messages = [flatten(p) for p in product(*FeaturesMatrices)]
        messages = filter(is_viable, messages)
        messages_dict = {from_vector_to_message(v): v for v in messages}

        return messages_dict

    def _get_semantics_dict(self, referents, messages):

        def is_square(n):
            return sqrt(abs(n)) == int(sqrt(abs(n)))

        semantics = {m: [] for m in messages}

        for mk, mv in messages.iteritems():

            for tk, tv in referents.iteritems():
                product = filter(None, map(lambda t, m: t * m, tv, mv))

                if len(product) == 3 and all([is_square(p) for p in product]):
                    semantics[mk].append(tk)

        return semantics

    def classify(self):
        level = {-1: [], 0: [], 1: [], 2: []}

        for ref_inst in self.reference_instances:
            level[ibr_classifier(ref_inst, arg_max=True)].append(ref_inst)

        self.unsolvable.extend(level[-1])
        self.level0.extend(level[0])
        self.level1.extend(level[1])
        self.level2.extend(level[2])


# 14-dimensional vectors
class RobustTuna:

    def __init__(self, corpus=CORPUS):
        self.games = []
        self.reference_instances = []
        self.unsolvable = []
        self.level0 = []
        self.level1 = []
        self.level2 = []

        self.TunaFeatures = [
            'chair', 'sofa', 'desk', 'fan',  # type
            'blue', 'red', 'green', 'grey',  # colour
            'left', 'right', 'front', 'back',  # orientation
            'large', 'small',  # size
            ]

        self.ComplexMessages = {}
        self.curate_complex_messages_dict()

        self.T = 7  # each context contains 1 target and 6 distractors
        self.F = 14

        self.gather_reference_instances()
        self.classify()

    def key(self, x):
        # indices of type, colour, orientation, and size
        return self.TunaFeatures.index(x)

    def curate_complex_messages_dict(self):

        def create_features_matrix(f):
            messages = [list(repeat(0, f)) for i in range(f + 1)]

            for i in range(f):
                messages[i][i] = 1

            return messages

        def flatten(li):
            return [i for feature in li for i in feature]

        def from_vector_to_message(v):
            message = [self.TunaFeatures[i] for i, w in enumerate(v) if w]

            return ' '.join(message)

        FeaturesMatrices = [
            create_features_matrix(4),  # type
            create_features_matrix(4),  # colour
            create_features_matrix(4),  # orientation
            create_features_matrix(2),  # size
            ]

        MESSAGES = [flatten(p) for p in product(*FeaturesMatrices)]

        for M in range(1, 5):
            messages = filter(lambda m: m.count(1) == M, MESSAGES)
            messages_dict = {from_vector_to_message(v): v for v in messages}
            self.ComplexMessages.update(messages_dict)

    def gather_reference_instances(self):
        for dirpath, dirname, filenames in os.walk(CORPUS):

            for f in filenames:
                filepath = dirpath + '/' + f
                ref_instances = list(self.create_reference_instance(filepath))
                ref_instances = filter(None, ref_instances)

                if ref_instances:
                    self.reference_instances.extend(ref_instances)

    def create_reference_instance(self, filepath):
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            # get target referent and create targets dictionary for Game
            target, referents = self._get_target_and_referents(root)

            # create all possible target messages
            possible_target_messages = self._get_target_messages(root)

            # create messages dictionary for Game
            messages = self.ComplexMessages

            # create semantics dictionary for Game
            semantics = self._get_semantics_dict(referents, messages)

            # create Game
            game = Game(targets=referents, messages=messages, sems=semantics)

            # add game to list of all games
            self.games.append(game)

            for message in possible_target_messages:

                # create reference instance
                ref_inst = {
                    'game': game,
                    'message': message,
                    'target': target,
                    }

                yield ref_inst

        # ignore non-xml files
        except ET.ParseError:
            pass

    def _get_target_and_referents(self, root):
        # create referents
        referents = {}

        for entity in root.iter('ENTITY'):
            referent = entity.attrib['ID']
            vector = list(repeat(0, self.F))

            attributes = entity.getchildren()

            # set target
            if entity.attrib['TYPE'] == 'target':
                target = referent

            # colour, type, orientation, and size attributes
            for attr in attributes[:4]:
                vector[self.key(attr.attrib['VALUE'])] = 1

            referents[referent] = vector

        return target, referents

    def _get_target_messages(self, root):
        possible_target_messages = []

        # type, colour, orientation, and size features
        entities = [e for e in root.iter('ENTITY')]
        attributes = [a.attrib['VALUE'] for a in entities[0].getchildren()[:4]]

        # messages can have a maximum length of 4 features
        for M in range(1, 5):

            # for every possible combination of features that is length M
            for c in combinations(attributes, M):
                c = ' '.join([a for a in sorted(c, key=self.key)])
                possible_target_messages.append(c)

        return possible_target_messages

    def _get_semantics_dict(self, referents, messages):
        semantics = {m: [] for m in messages}

        for mk, mv in messages.iteritems():
            M = mv.count(1)

            for tk, tv in referents.iteritems():
                if len(filter(None, map(lambda t, m: t * m, tv, mv))) == M:
                    semantics[mk].append(tk)

        return semantics

    def classify(self):
        level = {-1: [], 0: [], 1: [], 2: []}

        for ref_inst in self.reference_instances:
            level[ibr_classifier(ref_inst, arg_max=True)].append(ref_inst)

        self.unsolvable.extend(level[-1])
        self.level0.extend(level[0])
        self.level1.extend(level[1])
        self.level2.extend(level[2])


if __name__ == '__main__':
    robust = RobustTuna()
    set_trace()
