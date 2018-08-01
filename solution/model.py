import sys
import string
import json
import numpy as np

from keras.layers import Input, Dense, LSTM, Activation
from keras.models import Sequential, Model

# create the neural net
class MODEL():
    def __init__(self, dim, library_data, path_json):        
        self.n_char = 20
        self.encoding_len = 20
        
        # dict to map the circuit element to char and int
        # those mapping are needed for the neural net
        self.out_dim = 0
        self.circuit_to_char = None
        self.char_to_circuit = None
        self.char_to_indices = None
        self.indices_to_char = None
        
        self.characters = []
        self.get_out_dim(path_json)
        self.get_characters(library_data)
        self.get_mapping_dict()
        
        self.model = self.build_model()
        
    def get_characters(self, library_data):
        """
        function to get back the different element from
        the library.
        params - library_data: the library
        """
        for key in library_data['gates']:
            self.characters.append(key['id'])
            
    def get_out_dim(self, path_json):
        """ 
        function to get the number of elements
        we need to sample to test solutions to
        the problem.
        params - path_json: path to the json file
        of the circuit we want to run the algo on.
        """
        with open(path_json) as file:
            list_data = json.load(file) 
        
        out_dim = 0
        # we only look for NOT element as per
        # the problem description
        for gate in list_data['gates']:
            if gate['type'] == 'NOT':
                out_dim+=1

        self.out_dim = out_dim
    
    def get_mapping_dict(self):
        """
        function to fill the mapping dictionnary
        to go from char to int to circuit element
        """
        chars = list(string.ascii_lowercase)[:len(self.characters)]
        
        self.circuit_to_char = dict((c, i) for (c,i) in zip(self.characters, chars))
        self.char_to_circuit = dict((i, c) for (c,i) in zip(self.characters, chars))
        
        self.char_to_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_to_char = dict((i, c) for i, c in enumerate(chars))
        
    def build_model(self):
        """
        function to create the neural net
        """
        model = Sequential()
        model.add(LSTM(256))
        model.add(Dense(self.n_char))
        model.add(Activation('softmax'))

        return model
    
    def sample(self, preds, temperature=0.8):
        # helper function to sample an index from a probability array
        # comes from keras example
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
                
    def sample_solution(self):
        """
        function to generate a solution given
        a first token
        """
        # first token; not returned at the end
        generated = 'a'
        # create an array of the size of the number of element
        # we need to provide to solve the problem
        x_pred = np.zeros((1, self.out_dim+1, self.n_char))
        
        # we itratively grow our sequence by
        # predicting the next element
        for i in range(self.out_dim):
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds)          
            next_char = self.indices_to_char[next_index]
            
            generated += next_char
            x_pred[0,i+1,next_index] = 1
        
        return generated[1:]
    
    

   