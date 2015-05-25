import baseline_games
import ibr_classifier
import numpy

unsolvable, level0, level1, level2 = baseline_games.generate_classified_reference_instances()

def ibr_agents(game, message, target, depth=2):
	l0 = {}
	for m in game.messages.keys():
		l0[m] = {}
		for t in game.targets.keys():
			if t in game.sems[m]:
				l0[m][t] = 1.0/len(game.sems[m])
			else:
				l0[m][t] = 0
					
	messages_per_target = {}
	for t in game.targets.keys():
		messages_per_target[t] = 0
		for m in game.messages.keys():
			if t in game.sems[m]:
				messages_per_target[t] += 1
					
	s0 = {}
	for t in game.targets.keys():
		s0[t] = {}
		for m in game.messages.keys():
			if t in game.sems[m]:
				s0[t][m] = 1.0/messages_per_target[t]
			else:
				s0[t][m] = 0
	
	current_listener = l0
	current_speaker = s0
	current_depth = 0
	
	while current_depth != depth:
		current_listener, current_speaker = ibr_classifier.posterior(game,current_listener,current_speaker)
		current_depth += 1
	
	return current_listener, current_speaker

level0_dict = {}
level1_dict = {}
level2_dict = {}

for i in xrange (0, 20):
	level0_vals = []
	level1_vals = []
	level2_vals = []
	for instance in level0:
		game = instance['game']
		message = instance['message']
		target = instance['target']
		listener, speaker = ibr_agents(game, message, target, depth = i)
		level0_vals.append(listener[message][target])
	for instance in level1:
		game = instance['game']
		message = instance['message']
		target = instance['target']
		listener, speaker = ibr_agents(game, message, target, depth = i)
		level1_vals.append(listener[message][target])
	for instance in level2:
		game = instance['game']
		message = instance['message']
		target = instance['target']
		listener, speaker = ibr_agents(game, message, target, depth = i)
		level2_vals.append(listener[message][target])
	level0_dict[i] = numpy.mean(level0_vals)
	level1_dict[i] = numpy.mean(level1_vals)
	level2_dict[i] = numpy.mean(level2_vals)

print level0_dict
print level1_dict
print level2_dict
