import numpy as np
from distributedwordreps import ShallowNeuralNetwork
import random
import sys
from games import level0, level1, level2
from games import vogel_level0, vogel_level1, vogel_level2
from games import sep_games

class ANNListener(object):

    def __init__(self, games=None, hidden_dim=5, check_literal=None):
        self.games = games
        self.n_targets = len(games[0].targets)
        self.n_features = len(games[0].messages)

        self.check_literal = check_literal
        self.input_dim = self.n_targets * self.n_features + self.n_features
        if self.check_literal == "exist":
            # extra dimension to check whether literal interpretation is possible
            self.input_dim += 1
        elif self.check_literal == "all":
            # extra dimensions that encode the literal content
            self.input_dim += self.n_targets


        self.ann = ShallowNeuralNetwork(input_dim=self.input_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=self.n_features)
        self.role = "listener"

    def target_vec(self, game, message):
        targets = game.targets
        input_targets = sum(targets.values(), [])
        input_message = game.messages[message]

        input_ann = input_targets + input_message

        if self.check_literal == "exist":
            # extra dimension to check whether literal interpretation is possible
            input_literal = [1] if game.sems[message] else [0]
            input_ann += input_literal
        elif self.check_literal == "all":
            # extra dimensions that encode the literal content
            input_literal = [1 if target in game.sems[message] else 0 for target in targets]
            input_ann += input_literal

        return self.ann.predict(input_ann)

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
                if isinstance(message, list):
                    for msg in message:
                        input_ann = self.msg2input(game, msg, targets, input_targets)
                        output_ann = np.zeros(len(game.targets))
                        output_ann[indx] = 1.0
                        data.append((input_ann, output_ann))
                else:
                    input_ann = self.msg2input(game, message, targets, input_targets)
                    output_ann = np.zeros(len(game.targets))
                    output_ann[indx] = 1.0
                    data.append((input_ann, output_ann))

        if verbose:
            print "\nTraining the network..."
        self.ann.train(training_data=data, display_progress=verbose)
        return data

    def msg2input(self, game, message, targets, input_targets):
        input_message = game.messages[message]
        input_ann = input_targets + input_message

        if self.check_literal == "exist":
            # extra dimension to check whether literal interpretation is possible
            input_literal = [1] if game.sems[message] else [0]
            input_ann += input_literal
        elif self.check_literal == "all":
            # extra dimensions that encode the literal content
            input_literal = [1 if target in game.sems[message] else 0 for target in targets]
            input_ann += input_literal

        return input_ann

class ANNSpeaker(object):
    # there are two versions: one that represents the intended target
    # using its features (target_rep="features"), 
    # and one that represents the intended target using its position
    # (target_rep="position").

    def __init__(self, games=None, hidden_dim=5, 
                 target_rep="features", check_literal=None):
        self.games = games
        self.n_targets = len(games[0].targets)
        self.n_features = len(games[0].messages)
        self.target_rep = target_rep

        if target_rep == "features":
            self.input_dim = self.n_targets * self.n_features + self.n_features
        elif target_rep == "position":
            self.input_dim = self.n_targets * self.n_features + self.n_targets

        self.check_literal = check_literal
        if self.check_literal == "exist":
            # extra dimension to check whether there is any true literal message
            self.input_dim += 1
        elif self.check_literal == "all":
            # extra dimensions that encode whether each message is true
            self.input_dim += self.n_features

        self.ann = ShallowNeuralNetwork(input_dim=self.input_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=self.n_features)
        self.role = "speaker"

    def message_vec(self, game, target):
        targets = game.targets
        messages = game.messages
        input_targets = sum(targets.values(), [])

        if self.target_rep == "features":
            input_target_nodes = game.targets[target]
        elif self.target_rep == "position":
            input_target_nodes = [0 for i in range(len(targets))]
            indx = targets.keys().index(target)
            input_target_nodes[indx] = 1

        input_ann = input_targets + input_target_nodes
        if self.check_literal == "exist":
            # extra dimension to check whether there is any true literal message
            literal_messages = [m for m in game.messages if target in game.sems[m]]
            input_literal = [1] if literal_messages else [0]
            input_ann += input_literal
        elif self.check_literal == "all":
            # extra dimensions that encode whether each message is true
            input_literal = [1 if target in game.sems[message] else 0 for message in messages]
            input_ann += input_literal

        return self.ann.predict(input_ann)

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
                        if self.check_literal == "exist":
                            # extra dimension to check whether there is any true literal message
                            literal_messages = [m for m in game.messages if target in game.sems[m]]
                            input_literal = [1] if literal_messages else [0]
                            input_ann += input_literal
                        elif self.check_literal == "all":
                            # extra dimensions that encode whether each message is true
                            input_literal = [1 if target in game.sems[message] else 0 for message in messages]
                            input_ann += input_literal

                        output_ann = game.messages[message]
                        data.append((input_ann, output_ann))
        if verbose:
            print "\nTraining the network..."
        self.ann.train(training_data=data, display_progress=verbose)
        return data

class LitSpeaker(object):

    def __init__(self, games=None, produce_all=False):
        self.games = games
        self.role = "speaker"
        self.produce_all = produce_all

    def produce(self, game, target):
        # produces a message (from Messages dict)
        messages = game.messages.keys()

        if self.produce_all:
            return [message for message in messages if target in game.sems[message]]
        else:
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

def SelfTrain(games, max_iter = 10, verbose = True,
              speaker_hidden_dim=50, listener_hidden_dim=50,
              speaker_target_rep="position",
              chain_init="literal_speaker",
              store_eval=True,
              check_literal_L="all", check_literal_S=None,
              litS_produce_all=False):
    
    listener_evals_v = [0 for i in range(max_iter)]
    listener_evals_us = [0 for i in range(max_iter)]

    if verbose:
        print "initialize ANN listeners and speakers"
    listener = ANNListener(games, hidden_dim=listener_hidden_dim, 
                           check_literal=check_literal_L)
    speaker = ANNSpeaker(games, hidden_dim=speaker_hidden_dim, 
                         target_rep=speaker_target_rep, 
                         check_literal=check_literal_S)
    if verbose:
        print "done"

    if chain_init == "literal_speaker":
        literal_speaker = LitSpeaker(games, produce_all=litS_produce_all)
        if verbose:
            print "Iteration 1: start with literal speaker"

        listener.learn(literal_speaker, verbose)
        if store_eval:
            both_evals = SummaryEvaluateListener(listener)
            listener_evals_v[0] = both_evals[0]
            listener_evals_us[0] = both_evals[1]

        speaker.learn(listener, verbose)
        if verbose:
            print "done"

        n_iter = 1
        while n_iter < max_iter:
            if verbose:
                print "Iteration {0} out of {1}".format(n_iter + 1, max_iter)
            listener.learn(speaker, verbose)
            if store_eval:
                both_evals = SummaryEvaluateListener(listener)
                listener_evals_v[n_iter] = both_evals[0]
                listener_evals_us[n_iter] = both_evals[1]
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
        if store_eval:
            both_evals = SummaryEvaluateListener(listener)
            listener_evals_v[0] = both_evals[0]
            listener_evals_us[0] = both_evals[1]

        n_iter = 1
        if verbose:
            print "done"

        while n_iter < max_iter:
            if verbose:
                print "Iteration {0} out of {1}".format(n_iter + 1, max_iter)
            speaker.learn(listener, verbose)
            listener.learn(speaker, verbose)
            if store_eval:
                both_evals = SummaryEvaluateListener(listener)
                listener_evals_v[n_iter] = both_evals[0]
                listener_evals_us[n_iter] = both_evals[1]

            n_iter += 1
            if verbose:
                print "done"

    return [listener, speaker, listener_evals_v, listener_evals_us]

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

def SummaryEvaluateListener(listener, verbose=True):
    level0_accuracy_v = EvaluateListener(listener, vogel_level0)
    level1_accuracy_v = EvaluateListener(listener, vogel_level1)
    level2_accuracy_v = EvaluateListener(listener, vogel_level2)

    level0_accuracy_us = EvaluateListener(listener, level0)
    level1_accuracy_us = EvaluateListener(listener, level1)
    level2_accuracy_us = EvaluateListener(listener, level2)

    if verbose:
        print "Listener level0 accuracies: ",
        print "{0} of {1}".format(level0_accuracy_v, len(vogel_level0)),
        print " Us: {0} of {1}".format(level0_accuracy_us, len(level0))
        print "Listener level1 accuracies: ",
        print "{0} of {1}".format(level1_accuracy_v, len(vogel_level1)),
        print " Us: {0} of {1}".format(level1_accuracy_us, len(level1))
        print "Listener level2 accuracies: ",
        print "{0} of {1}".format(level2_accuracy_v, len(vogel_level2)),
        print " Us: {0} of {1}".format(level2_accuracy_us, len(level2))

    return [[level0_accuracy_v, level1_accuracy_v, level2_accuracy_v],
            [level0_accuracy_us, level1_accuracy_us, level2_accuracy_us]]
    
