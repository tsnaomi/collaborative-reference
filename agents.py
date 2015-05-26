import numpy as np
from distributedwordreps import ShallowNeuralNetwork
import random
import sys
from games import level0, level1, level2

class ANNListener(object):

    def __init__(self, games=None, hidden_dim=5):
        self.games = games
        self.n_targets = len(games[0].targets)
        self.n_features = len(games[0].messages)
        self.input_dim = self.n_targets * self.n_features + self.n_features
        self.ann = ShallowNeuralNetwork(input_dim=self.input_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=self.n_features)
        self.role = "listener"

    def target_vec(self, game, message):
        targets = game.targets
        input_targets = sum(targets.values(), [])
        input_message = game.messages[message]

        return self.ann.predict(input_targets + input_message)

    def interpret(self, game, message):
        # produces a target (from Targets dict)
        targets = game.targets
        target_vec = self.target_vec(game, message)

        # find the maximum score in target_vec; return the corresponding target
        target_index = target_vec.argmax()

        return (targets.keys()[target_index])


    def learn(self, speaker, verbose=True):
        data = []
        N = len(self.games)
        for n, game in enumerate(self.games):
            if verbose:
                print "\rGenerating training data ({0} of {1})".format(n+1, N),
                sys.stdout.flush()
            targets = game.targets
            input_targets = sum(targets.values(), [])

            for indx, target in enumerate(targets.keys()):
                message = speaker.produce(game, target)
                input_message = game.messages[message]
                input_ann = input_targets + input_message
                output_ann = np.zeros(len(game.targets))
                output_ann[indx] = 1.0
                data.append((input_ann, output_ann))
        if verbose:
            print "\nTraining the network..."
        self.ann.train(training_data=data, display_progress=verbose)
        return data

class ANNSpeaker(object):
    # there could be two versions, one that represents the intended target
    # using its features (target_rep="features"), 
    # and one that represents the intended target using its position
    # (target_rep="position").

    def __init__(self, games=None, hidden_dim=5, target_rep="features"):
        self.games = games
        self.n_targets = len(games[0].targets)
        self.n_features = len(games[0].messages)
        self.target_rep = target_rep

        if target_rep == "features":
            self.input_dim = self.n_targets * self.n_features + self.n_features
        elif target_rep == "position":
            self.input_dim = self.n_targets * self.n_features + self.n_targets

        self.ann = ShallowNeuralNetwork(input_dim=self.input_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=self.n_features)
        self.role = "speaker"

    def message_vec(self, game, target):
        targets = game.targets
        input_targets = sum(targets.values(), [])

        if self.target_rep == "features":
            input_target_nodes = game.targets[target]
        elif self.target_rep == "position":
            input_target_nodes = [0 for i in range(len(targets))]
            indx = targets.keys().index(target)
            input_target_nodes[indx] = 1

        return self.ann.predict(input_targets + input_target_nodes)

    def produce(self, game, target):
        # produces a target (from Targets dict)
        messages = game.messages
        message_vec = self.message_vec(game, target)

        # find the maximum score in message_vec; return the corresponding message
        message_index = message_vec.argmax()

        for message, vec in messages.items():
            if vec[message_index] == 1:
                return message


    def learn(self, listener, verbose=True):
        data = []
        N = len(self.games)
        for n, game in enumerate(self.games):
            if verbose:
                print "\rGenerating training data ({0} of {1})".format(n+1, N),
                sys.stdout.flush()

            messages = game.messages
            targets = game.targets
            input_targets = sum(targets.values(), [])

            for target in targets.keys():
                for message in messages.keys():
                    if target == listener.interpret(game, message):
                        if self.target_rep == "features":
                            input_target_nodes = targets[target]
                        elif self.target_rep == "position":
                            input_target_nodes = [0 for i in range(len(targets))]
                            indx = targets.keys().index(target)
                            input_target_nodes[indx] = 1 

                        input_ann = input_targets + input_target_nodes
                        output_ann = game.messages[message]
                        data.append((input_ann, output_ann))
        if verbose:
            print "\nTraining the network..."
        self.ann.train(training_data=data, display_progress=verbose)
        return data

class LitSpeaker(object):

    def __init__(self, games=None):
        self.games = games
        self.role = "speaker"

    def produce(self, game, target):
        # produces a message (from Messages dict)
        messages = game.messages.keys()
        random.shuffle(messages)
        for message in messages:
            if target in game.sems[message]:
                return message

        return messages[0]

class LitListener(object):

    def __init__(self, games=None):
        self.games = games
        self.role = "listener"

    def interpret(self, game, message):
        # produces a target (from Targets dict)
        targets = game.targets.keys()
        random.shuffle(targets)
        for target in targets:
            if target in game.sems[message]:
                return target

        return targets[0]

def SelfTrain(games, max_iter = 2, verbose = True,
              speaker_hidden_dim=10, listener_hidden_dim=10,
              speaker_target_rep="features",
              chain_init="literal_speaker"):
    if verbose:
        print "initialize ANN listeners and speakers"
    listener = ANNListener(games, hidden_dim=listener_hidden_dim)
    speaker = ANNSpeaker(games, hidden_dim=speaker_hidden_dim, 
                         target_rep=speaker_target_rep)
    if verbose:
        print "done"

    if chain_init == "literal_speaker":
        literal_speaker = LitSpeaker(games)
        if verbose:
            print "Iteration 1: start with literal speaker"

        listener.learn(literal_speaker, verbose)
        speaker.learn(listener, verbose)
        if verbose:
            print "done"

        n_iter = 1
        while n_iter < max_iter:
            if verbose:
                print "Iteration {0} out of {1}".format(n_iter + 1, max_iter)
            listener.learn(speaker, verbose)
            speaker.learn(listener, verbose)
            n_iter += 1
            if verbose:
                print "done"

    if chain_init == "literal_listener":
        if verbose:
            print "Iteration 1: start with literal listener"
        literal_listener = LitListener(games)
        speaker.learn(literal_listener, verbose)
        listener.learn(speaker, verbose)
        n_iter = 1
        if verbose:
            print "done"

        while n_iter < max_iter:
            if verbose:
                print "Iteration {0} out of {1}".format(n_iter + 1, max_iter)
            speaker.learn(listener, verbose)
            listener.learn(speaker, verbose)
            n_iter += 1
            if verbose:
                print "done"

    return [listener, speaker]

def InspectSpeaker(speaker, N=-1):
    if N == -1:
        games = speaker.games
        N = len(games)
    else:
        games = speaker.games[:N]

    for n, game in enumerate(games):
        print "Game {0} out of {1}:".format(n+1, N)
        targets = game.targets
        for target in targets:
            message = speaker.produce(game, target)
            truth = target in game.sems[message]
            print ">> " + target + ": " +  message + " ({0})".format(truth)

    return 0

def InspectListener(listener, N=-1):
    if N == -1:
        games = listener.games
        N = len(games)
    else:
        games = listener.games[:N]

    for n, game in enumerate(games):
        print "Game {0} out of {1}:".format(n+1, N)
        messages = game.messages
        for message in messages:
            target = listener.interpret(game, message)
            truth = target in game.sems[message]
            print ">> " + message + ": " +  target + " ({0})".format(truth)

    return 0

def Inspect(agent, N=-1):
    if agent.role == "listener":
        return InspectListener(agent, N=N)
    if agent.role == "speaker":
        return InspectSpeaker(agent, N=N)

def EvaluateListener(listener, refs):
    # evaluate listener on the set of refs 
    N = len(refs)
    correct = 0.0
    for ref in refs:
        if listener.interpret(ref["game"], ref["message"]) == ref["target"]:
            correct += 1
    
    return correct / N

def SummaryEvaluateListener(listener):
    print "Listener accuracies on level0 instances: ",
    print "{0} of {1}".format(EvaluateListener(listener, level0), len(level0))
    print "Listener accuracies on level1 instances: ",
    print "{0} of {1}".format(EvaluateListener(listener, level1), len(level1))
    print "Listener accuracies on level2 instances: ",
    print "{0} of {1}".format(EvaluateListener(listener, level2), len(level2))
    
