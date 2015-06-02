import unittest

from ibr_classifier import ibr_classifier
from games import Game


reference_instances = [
    (0, {
        'game': Game(
            messages={
                'hats': [1, 0, 0],
                'mustache': [0, 0, 1],
                'glasses': [0, 1, 0],
                },
            targets={
                'right': [0, 1, 1],
                'center': [0, 0, 1],
                'left': [1, 1, 0],
                },
            sems={
                'hats': ['left'],
                'mustache': ['center', 'right'],
                'glasses': ['left', 'right'],
                }
            ),
        'message': 'hats',
        'target': 'left',
        }),
    (1, {
        'game': Game(
            messages={
                'hats': [1, 0, 0],
                'mustache': [0, 0, 1],
                'glasses': [0, 1, 0],
                },
            targets={
                'right': [0, 1, 1],
                'center': [0, 0, 1],
                'left': [1, 1, 0],
                },
            sems={
                'hats': ['left'],
                'mustache': ['center', 'right'],
                'glasses': ['left', 'right'],
                }
            ),
        'message': 'mustache',
        'target': 'center',
        }),
    (2, {
        'game': Game(
            messages={
                'hats': [1, 0, 0],
                'mustache': [0, 0, 1],
                'glasses': [0, 1, 0],
                },
            targets={
                'right': [0, 1, 1],
                'center': [0, 0, 1],
                'left': [1, 1, 0],
                },
            sems={
                'hats': ['left'],
                'mustache': ['center', 'right'],
                'glasses': ['left', 'right'],
                }
            ),
        'message': 'glasses',
        'target': 'right',
        }),
    (2, {
        'game': Game(
            messages={
                'hats': [1, 0, 0],
                'glasses': [0, 1, 0],
                'mustache': [0, 0, 1],
                },
            targets={
                'left': [0, 0, 0],
                'center': [0, 0, 1],
                'right': [0, 0, 1],
                },
            sems={
                'hats': [],
                'glasses': [],
                'mustache': ['center', 'right', ],
                },
            ),
        'message': 'hats',
        'target': 'left',
        }),
    ]


class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.reference_instances = reference_instances

    def test_classifier(self):
        for gold, ref_inst in self.reference_instances:
            test = ibr_classifier(**ref_inst)
            self.assertEqual(
                test,
                gold,
                msg='\n\n%s\n\nTest: %s\nGold: %s' % (ref_inst, test, gold),
                )


if __name__ == '__main__':
    unittest.main()
