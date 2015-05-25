def ibr_classifier(game, message, target):
	if target not in game.sems[message]:
		return -1
	else:
		level_zero = {}
		for m in game.messages.keys():
			level_zero[m] = {}
			for t in game.targets.keys():
				if target in game.sems[message]:
					level_zero[m][t] = 1.0/len(game.sems[message])
				else:
					level_zero[m][t] = 0
		level_one = posterior(game,level_zero)
		level_two = posterior(game,level_one)
		if is_solved(level_zero,message,target):
			return 0
		elif is_solved(level_one,message,target):
			return 1
		elif is_solved(level_two,message,target):
			return 2
		else:
			return -1

def is_solved(ibr_dict, message, target):
	target_val = ibr_dict[message][target]
	solved = True
	for t in ibr_dict[message]: 
		if t!= target and level_one[message][t] >= target_val:
			solved = False
	return solved
			
def posterior(game,ibr_dict):
	targets = game.targets.keys()
	messages = game.messages.keys()
	
	new_ibr_dict = {}
	target_priors = {}
	target_probs = {}
	
	for t in targets:
		target_priors[t] = 1.0/len(targets)
	
	for t in targets:
		target_probs[t] = 0
		for m in messages:
			target_probs[t] += ibr_dict[m][t]
	
	for m in messages:
		sum = 0.0
		for t in targets:
			sum += target_priors[t] * ibr[m][t] / target_probs[t]
		for t in targets:	
			new_ibr_dict[m][t] = target_priors[t] * (ibr_dict[m][t] / target_probs[t]) / sum 
			
	return new_ibr_dict