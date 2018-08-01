import sys
import json
import argparse
import pickle
import time
import numpy as np

import model as m
from evostra import EvolutionStrategy

sys.path.append('../')
from genetic_circuit_scoring import CircuitMapping

parser = argparse.ArgumentParser()
parser.add_argument('--path_json', type=str, help='Path to the json file')
parser.add_argument('--path_library', type=str, help='Path to the  library')
parser.add_argument('--name', type=str, help='Name to save the dict of your run')
parser.add_argument('--n_epoch', type=int, help='The number of epoch to run the algorithm')
args = parser.parse_args()

start = time.time()

# We define global var to
# keep track of the best results
# This is done because the library
# used does not return the best 
# intermediary results
global best
best = 0
global best_circuit

with open(args.path_library) as library_file:
    library_data = json.load(library_file)
    

def get_score(n_samples, weights):
    """
    function to get the score of a given seqence
    of elements given by the algo
    params – n_samples: number of samples sampled
    by the algo for this run
    params - weights: weights of the neural net
    """
    all_scores = []
    for i in range(n_samples):
        
        #load the current weights and sample solution
        net.model.set_weights(weights)
        solution = net.sample_solution()
        circuit = [net.char_to_circuit[str(x)] for x in solution]
        score = get_score_mapping(circuit, args.path_json)
        
        # update the status of the current best
        #solution
        global best
        global best_circuit
        if score > best:
            best = score 
            best_circuit = circuit
        all_scores.append(score)
    
    # return the maximum of all the samples
    # because we aim for the max score
    # note: we could have optimise for the mean
    # of all our samples of a given epoch e.g.
    return np.amax(all_scores)

def get_reward(weights):
    """
    function needed by evostra (ES library)
    to optomise the weights of a neural net
    given a number to optimise, i.e. the reward
    params - weights: parameters of the neural nets,
    i.e. its weights
    """
    # n_samples is the number of samples 
    # sampled at each epoch
    n_samples = 1000
    return get_score(n_samples, weights)

def get_score_mapping(mapping_from_net, path):
    """
    function to get the score back from generated
    outpout of the neural net.
    params - mapping_from_net: output from the neural net
    params - path: path to the json file
    """
        
    # we load the original file
    # at every call to avoid
    # messing up with cloning
    # of nested objects
    with open(path) as file:
        list_data = json.load(file)  
    
    # idx_map is the idx to keep track
    # of where we have to add mapping
    # i.e. for NOT element only
    idx_map = 0
    for i,gate in enumerate(list_data['gates']):
        if gate['type'] == 'NOT':
            gate['mapping'] = mapping_from_net[idx_map]
            idx_map +=1
    
    # we use try to return the score
    # if the circuit was valid according
    # to circuit_mapping
    # return 0 if circuit_mapping throw
    # an error
    try:
        circuit_mapping.map(list_data)
        return circuit_mapping.score()
    except:
        return 0
    
def save_dict(path, name, verbose=True):
    """
    function to save the best solution at the end of the
    algo.
    params - path: path of the json file we want to
    run the algo on.
    params - name_ name of the file that will be saved, i.e.
    the resulting best dictionnary.
    params - verbose: if set to True, we display in the terminal
    the best results and the element chosen.
    """
    global best_circuit
    global best
   
    if verbose:
        print(f'best score: {best}')
        print(f'elements used: {best_circuit}')
    
    with open(path) as file:
        list_data = json.load(file)  
    
    # Modify the file with the best
    # mapping found
    idx_map = 0
    for i,gate in enumerate(list_data['gates']):
        if gate['type'] == 'NOT':
            gate['mapping'] = best_circuit[idx_map]
            idx_map +=1
    
    # we save the score along with the name of the run
    score_str = str(round(best, 2))
    with open(f'{name}_{score_str}.pickle', 'wb') as handle:
        pickle.dump(list_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# create the circuit_mapping object   
circuit_mapping = CircuitMapping(library_data)
# create the neural net
net = m.MODEL(20, library_data, args.path_json)

# run the evolution strategies (ES)
# note: if you want, you can play with the paramters.
# those seems to give reasonable results
es = EvolutionStrategy(net.model.get_weights(), 
                       get_reward, 
                       population_size=5, 
                       sigma=0.01,  # noise std deviation
                       learning_rate=0.001, 
                       decay=0.995,
                       num_threads=1)

es.run(args.n_epoch)
save_dict(args.path_json, args.name, verbose=True)
done = time.time()
elapsed = done - start
print(f'elapsed time: {elapsed}')




    

