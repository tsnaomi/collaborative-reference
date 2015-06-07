import numpy as np
import copy
from collections import *

Agent = namedtuple('Agent', ['mat','colnames','rownames'])
Game = namedtuple('Game', ['messages', 'targets', 'sems'])

def ibr_classifier (instance, max_depth = 2, arg_max=False, starting_agent = None, fg = False, display = False):
	game = instance['game']
	sems = game.sems
	inverse = inverse_sems(game)
	messages = game.messages.keys()
	targets = game.targets.keys()
	
	dim1 = len(messages)
	dim2 = len(targets)
	mat_size = float(dim1 * dim2)
	
	l0 = Agent(mat=np.arange(mat_size).reshape(dim1,dim2), colnames = targets, rownames = messages)

	for i in xrange(0,len(l0.rownames)):
		for j in xrange(0,len(l0.colnames)):
			m = l0.rownames[i]
			t = l0.colnames[j]
			l0.mat[i][j] = 1.0/len(sems[m]) if t in sems[m] else 0
		
	for i in xrange(0,len(l0.rownames)): 
		if np.sum(l0.mat[i]) == 0:
			for j in xrange(0,len(l0.colnames)):
				l0.mat[i][j] = 1.0/len(l0.colnames)
	
	if not fg:
		s0 = Agent(mat=np.arange(mat_size).reshape(dim2,dim1), colnames = messages, rownames = targets)
	
		for i in xrange(0,len(s0.rownames)):
			for j in xrange(0,len(s0.colnames)):
				t = s0.rownames[i]
				m = s0.colnames[j]
				s0.mat[i][j] = 1.0/len(inverse[t]) if m in inverse[t] else 0
			
		for i in xrange(0,len(s0.rownames)): 
			if np.sum(s0.mat[i]) == 0:
				for j in xrange(0,len(s0.colnames)):
					s0.mat[i][j] = 1.0/len(s0.colnames)
	
	else:
		starting_agent = 'speaker'
		fg_list = Agent(mat=np.arange(mat_size).reshape(dim2,dim1), colnames = targets, rownames = messages)
	
		for i in xrange(0,len(fg_list.rownames)):
			for j in xrange(0,len(fg_list.colnames)):
				m = fg_list.rownames[i]
				t = fg_list.colnames[j]
				sum = 0.0
				for w in inverse[t]:
					sum += 1.0/len(sems[w])
				fg_list.mat[i][j] = (1.0/len(sems[m]))/sum if t in sems[m] else 0
			
		for i in xrange(0,len(fg_list.rownames)): 
			if np.sum(fg_list.mat[i]) == 0:
				for j in xrange(0,len(fg_list.colnames)):
					fg_list.mat[i][j] = 1.0/len(fg_list.colnames)

		s0 = bayes(fg_list)
		
	if arg_max:
		l0 = argmax(l0)
		s0 = argmax(s0)
	
	if display:
		print 'S0: \n'
		print s0
		print '================================================================'
		print 'L0: \n'
		print l0
		print '================================================================'
	
	if is_solved(instance, l0):
		return 0
	
	elif max_depth > 0:
		depth = 1
		
		current_listener = l0
		current_speaker = s0
		
		while depth <= max_depth:
			if not starting_agent: 
				new_speaker = bayes(current_listener)
				new_listener = bayes(current_speaker)
			
			elif starting_agent == 'listener':
				new_speaker = bayes(current_listener)
				new_listener = bayes(new_speaker)
				
			elif starting_agent == 'speaker':
				new_listener = bayes(current_speaker)
				new_speaker = bayes(new_listener)
				
			else:
				raise NameError('Invalid starting agent argument.')
			
			if arg_max:
				new_speaker = argmax(new_speaker)
				new_listener = argmax(new_listener)
			
			if display:
				print 'S%d: \n' % depth
				print new_speaker
				print '================================================================'
				print 'L%d: \n' % depth
				print new_listener
				print '================================================================'
			
			if is_solved(instance, new_listener):
				return depth
			
			current_speaker = new_speaker
			current_listener = new_listener
			
			depth += 1
	
	return -1
	
def inverse_sems(game):
	messages = game.messages.keys()
	targets = game.targets.keys()
	
	inverse_sems = {}
	
	for t in targets:
		inverse_sems[t] = []
		for m in messages:
			if t in game.sems[m]:
				inverse_sems[t].append(m)
	
	return inverse_sems

def argmax(old_agent):
	old_agent = copy.deepcopy(old_agent)
	new_agent = Agent(mat = old_agent.mat, colnames = old_agent.colnames, rownames = old_agent.rownames)
	
	for row in new_agent.mat:
		max = np.amax(row)
		max_positions = [i for i, j in enumerate(row) if j == max]
		if len(max_positions) == 1:
			for i in xrange(0, len(new_agent.colnames)):
				row[i] = 0
			row[max_positions[0]] = 1
	
	return new_agent

def bayes (old_agent):
	old_agent = copy.deepcopy(old_agent)
	new_agent = Agent(mat=np.transpose(old_agent.mat),colnames = old_agent.rownames,rownames = old_agent.colnames)
	
	for i in xrange(0,len(new_agent.rownames)):
		new_agent.mat[i] = new_agent.mat[i]/np.sum(new_agent.mat[i]) if np.sum(new_agent.mat[i]) != 0 else 0
	
	for i in xrange(0,len(new_agent.rownames)): 
		if np.sum(new_agent.mat[i]) == 0:
			for j in xrange(0,len(new_agent.colnames)):
				new_agent.mat[i][j] = 1.0/len(new_agent.colnames)
	
	return new_agent

def is_solved (instance, listener):
	message = instance['message']
	target = instance['target']
	
	return listener.mat[listener.rownames.index(message)][listener.colnames.index(target)] > np.amax(np.delete(listener.mat[listener.rownames.index(message)],listener.colnames.index(target)))