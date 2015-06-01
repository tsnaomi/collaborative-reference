import unittest

from ibr_classifier import ibr_classifier
from games import Game

reference_instances = [
    ({
        'game': Game(
            {'hats': [1, 0, 0], 'glasses': [0, 1, 0], 'mustache': [0, 0, 1]},
            {'left': [0, 0, 0], 'center': [0, 0, 1], 'right': [0, 0, 1]},
            {'hats': [], 'glasses': [], 'mustache': ['center', 'right', ]},
            ),
        'message': 'hats',
        'target': 'left',
        }, 2),

    ]


class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.reference_instances = reference_instances

    def test_classifier(self):
        for ref_inst, gold in self.reference_instances:
            test = ibr_classifier(**ref_inst)
            self.assertEqual(
                test,
                gold,
                msg='%s\nTest: %s\nGold: %s' % (str(ref_inst), test, gold),
                )


if __name__ == '__main__':
    unittest.main()
