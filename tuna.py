import os
import xml.etree.ElementTree as ET

from games import Game
from ibr_classifier import ibr_classifier
from itertools import combinations, product, repeat
from pdb import set_trace
from pprint import pprint


CORPUS = 'TUNA/corpus/singular/furniture'


# 14-dimensional vectors
class Tuna:

    def __init__(self, corpus=CORPUS):
        self.games = []

        # this contains every possible reference instance given the
        # game contexts provided in TUNA and every possible message
        # that is true of the target referent
        self.reference_instances = []
        self.unsolvable = []
        self.level0 = []
        self.level1 = []
        self.level2 = []

        # this contains only the reference instances provided in TUNA
        self.true_reference_instances = []
        self.true_unsolvable = []
        self.true_level0 = []
        self.true_level1 = []
        self.true_level2 = []

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

    def key(self, x):
        # indices of type, colour, orientation, and size
        return self.TunaFeatures.index(x)

    def curate_complex_messages_dict(self):
        # this functin curates self.ComplexMessages, a dictionary that contains
        # every possible message of length 1 through 4 given the features in
        # self.TunaFeatures

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
        # this function combs through TUNA and collects all of its games and
        # possible reference instance, extending self.reference_instances and
        # self.true_reference_instances through create_reference_instances()
        for dirpath, dirname, filenames in os.walk(CORPUS):

            for f in filenames:
                filepath = dirpath + '/' + f
                self.create_reference_instances(filepath)

    def create_reference_instances(self, filepath):
        # given an xml file in TUNA, this function produces the game and every
        # possible reference instance in that file, extending
        # self.reference_instances, self.true_references_instances, etc.
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            # ignore games in which location is necessary to identify target
            TRIAL = [t for t in root.iter('TRIAL')][0]
            if TRIAL.attrib['CONDITION'] == '-LOC':

                # get target referent and create targets dictionary for Game
                target, referents = self._get_target_and_referents(root)

                # create all possible target messages
                possible_target_messages = self._get_target_messages(root)

                # create messages dictionary for Game
                messages = self.ComplexMessages

                # create semantics dictionary for Game
                semantics = self._get_semantics_dict(referents, messages)

                # create Game
                game = Game(
                    targets=referents,
                    messages=messages,
                    sems=semantics,
                    )

                # get true complex message from xml file
                true_message = self._get_true_complex_message(root)

                # add game to list of all games
                self.games.append(game)

                for message in possible_target_messages:

                    # create reference instance
                    ref_inst = {
                        'game': game,
                        'message': message,
                        'target': target,
                        }

                    # classify ref_inst with ibr_classifier and add to either
                    # self.unsolvable, self.level0, self.level1, etc.,
                    # depending on its classification
                    try:
                        level = ibr_classifier(ref_inst, arg_max=True)
                        level = 'level%s' % level
                        getattr(self, level).append(ref_inst)

                    except AttributeError:
                        self.unsolvable.append(ref_inst)

                    # add ref_inst to self.reference_instances
                    self.reference_instances.append(ref_inst)

                    # curate true reference instances
                    if message == true_message:

                        # add ref_inst to self.reference_instances
                        self.true_reference_instances.append(ref_inst)

                        # add ref_inst to either self.true_unsolvable,
                        # self.true_level0, self.true_level1, etc., depending
                        # on its ibr classification
                        try:
                            getattr(self, 'true_%s' % level).append(ref_inst)

                        except AttributeError:
                            self.true_unsolvable.append(ref_inst)

        # ignore non-xml files
        except ET.ParseError:
            pass

    def _get_true_complex_message(self, root):

        def non_dimension(a):
            return a.attrib['NAME'] not in 'x-dimension y-dimension'

        # given a TUNA file, grab and normalize the message produced by an
        # actual speaker (located at the bottom of each xml file)
        attr_set = root.find('ATTRIBUTE-SET').getchildren()
        attr_set = [a.attrib['VALUE'] for a in attr_set if non_dimension(a)]
        attr_set = filter(lambda a: a not in 'other unknown', attr_set)
        message = ' '.join([a for a in sorted(attr_set, key=self.key)])

        return message

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
        entities = [e for e in root.iter('ENTITY')][0]
        attributes = [a.attrib['VALUE'] for a in entities.getchildren()[:4]]

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


if __name__ == '__main__':
    tuna = Tuna()
    set_trace()
