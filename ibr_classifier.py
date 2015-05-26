def ibr_classifier(game, message, target, depth=2):
    if target not in game.sems[message]:
        return -1
    else:
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
                    
        if is_solved(l0,message,target):
            return 0
        
        else:
        
            current_listener = l0
            current_speaker = s0
            current_depth = 1
        
            while current_depth <= depth:
                new_listener, new_speaker = posterior(game,current_listener,current_speaker)
                
                if is_solved(new_listener,message,target):
                    return current_depth
                
                else:
                    current_listener = new_listener
                    current_speaker = new_speaker
                    current_depth += 1
                    
            return -1

def is_solved(ibr_dict, message, target):
    target_val = ibr_dict[message][target]
    solved = True
    for t in ibr_dict[message]: 
        if t!= target and ibr_dict[message][t] >= target_val:
            solved = False
    return solved
            
# listener_dict = {'all':{'1':0.0,'2':1.0},'some':{'1':0.5,'2':0.5}}
# speaker_dict = {'1':{'all':0.0,'some':1.0},'2':{'all':0.5,'some':0.5}}        
# target_probs = {'1':0.5,'2':0.5}    
# message_probs = {'all':0.5,'some':0.5}
            
def posterior(game, listener_dict, speaker_dict,target_probs=None,message_probs=None):
    targets = game.targets.keys()
    messages = game.messages.keys()
    
    new_listener_dict = {}
    new_speaker_dict = {}
    
    if not target_probs:    # Assume uniform prior distribution unless otherwise specified
        target_probs = {}
        for t in targets:
            target_probs[t] = 1.0/len(targets)
    
    if not message_probs:    # Assume uniform prior distribution unless otherwise specified
        message_probs = {}
        for m in messages:
            message_probs[m] = 1.0/len(messages)
    
    for m in messages:
        new_listener_dict[m] = {}
        sum = 0.0
        for t in targets:
            sum += target_probs[t] * speaker_dict[t][m]
        for t in targets:
            new_listener_dict[m][t] = target_probs[t] * speaker_dict[t][m]/sum if sum else 0
        
    for t in targets:
        new_speaker_dict[t] = {}
        sum = 0.0
        for m in messages:
            sum += message_probs[m] * listener_dict[m][t]
        for m in messages:
            new_speaker_dict[t][m] = message_probs[m] * listener_dict[m][t]/sum if sum else 0
            
    return (new_listener_dict,new_speaker_dict)

