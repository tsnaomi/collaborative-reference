import copy
import itertools
import random
from collections import namedtuple
from numpy import dot, outer

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
    'hat': [1, 0, 0],
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
        if games == None:
        	self.type = 'literal'
        else:
        	self.type = 'neural'
        	self.net = self.learn()
    
    def is_literal(self):
    	return self.type == 'literal'

    def produce(game, target):
        # produces a message (from Messages dict)
        if self.is_literal():
        	target_features = game['sems'][target]
        	possible_messages = []
        	if target_features[0]:
        		possible_messages.append('hat')
        	if target_features[1]:
        		possible_messages.append('glasses')
        	if target_features[2]:
        		possible_messages.append('mustache')
        	return random.choice(possible_messages)
        else:
        	prediction_vector = self.net.predict(game['sems'] + Targets[target])
        	max = prediction_vector.indexOf(max(prediction_vector))
        	if max == 0:
        		return 'hat'
        	elif max == 1:
        		return 'glasses'
        	else:
        		return 'mustache'

    def learn():
        net = ShallowNeuralNetwork(input_dim = 12, hidden_dim = 10, output_dim = 3)
        training_data = []
        for game in games:
        	training_data.append((game['sems'],game['messages']))
        net.train(training_data =  , maxiter = 1000, display_progress = True)
        return net

def play(game):
    pass

class ShallowNeuralNetwork:
    def __init__(self, input_dim=0, hidden_dim=0, output_dim=0, afunc=np.tanh, d_afunc=(lambda z : 1.0 - z**2)):        
        self.afunc = afunc 
        self.d_afunc = d_afunc      
        self.input = np.ones(input_dim+1)   # +1 for the bias                                         
        self.hidden = np.ones(hidden_dim+1) # +1 for the bias        
        self.output = np.ones(output_dim)        
        self.iweights = randmatrix(input_dim+1, hidden_dim)
        self.oweights = randmatrix(hidden_dim+1, output_dim)        
        self.oerr = np.zeros(output_dim+1)
        self.ierr = np.zeros(input_dim+1)
        
    def forward_propagation(self, ex):        
        self.input[ : -1] = ex # ignore the bias
        self.hidden[ : -1] = self.afunc(dot(self.input, self.iweights)) # ignore the bias
        self.output = self.afunc(dot(self.hidden, self.oweights))
        return copy.deepcopy(self.output)
        
    def backward_propagation(self, labels, alpha=0.5):
        labels = np.array(labels)       
        self.oerr = (labels-self.output) * self.d_afunc(self.output)
        herr = dot(self.oerr, self.oweights.T) * self.d_afunc(self.hidden)
        self.oweights += alpha * outer(self.hidden, self.oerr)
        self.iweights += alpha * outer(self.input, herr[:-1]) # ignore the bias
        return np.sum(0.5 * (labels-self.output)**2)

    def train(self, training_data, maxiter=5000, alpha=0.05, epsilon=1.5e-8, display_progress=False):       
        iteration = 0
        error = sys.float_info.max
        while error > epsilon and iteration < maxiter:            
            error = 0.0
            random.shuffle(training_data)
            for ex, labels in training_data:
                self.forward_propagation(ex)
                error += self.backward_propagation(labels, alpha=alpha)           
            if display_progress:
                print 'completed iteration %s; error is %s' % (iteration, error)
            iteration += 1
                    
    def predict(self, ex):
        self.forward_propagation(ex)
        return copy.deepcopy(self.output)
        
    def hidden_representation(self, ex):
        self.forward_propagation(ex)
        return self.hidden