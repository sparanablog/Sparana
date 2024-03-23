""" These are scripts to track the data as it passes through a network. I am trying to track changes in the cosine distances between datapoints at each layer."""

import numpy as np
import cupy as cp
from scipy.spatial import distance

def get_cosines():
    print('thing')
    return

class distance_tracker:
    
    def __init__(self, model):
        self._model = model
        # Initialise arrays to store distances class at each layer
        self._cosines = []
        return
    
    def checkpoint(self, inputs, labels):
        # Get outputs of each layer, run the thing
        cosines = []
        outputs = self._model.outputs(inputs)
        #loop through the layers
        for layer in self._model.layers:
            these_cosines = np.zeros((labels.shape[1], labels.shape[1]))
            these_datapoints = np.zeros((labels.shape[1], labels.shape[1]))
            # Loop through the datapoints and take the cosines
            for i in range(len(inputs)):
                for j in range(len(inputs)-i):
                    # The indices to use are i and i+j Need to convert to numpy
                    cosine = distance.cosine(cp.asnumpy(layer._outputs[i]), cp.asnumpy(layer._outputs[i+j]))
                    # Need to put them in the upper triangle of the matrix
                    indices = tuple(sorted((np.argmax(labels[i]), np.argmax(labels[i+j]))))
                    these_cosines[indices] += cosine
                    these_datapoints[indices] += 1
            these_datapoints[these_datapoints == 0] = 1
            cosines.append(these_cosines / these_datapoints)
        self._cosines.append(cosines)
        return
    
    def return_some_values(self):
        """gets somm averages"""
        
        return
    
    def get_cosines(self, index = None):
        """ Returns consine distances between datapoints of different classes. 
        If an index is chosen, that will be the only class, if not then all classes will be averaged.
        Index needs to be a tuple (x,y)"""
        cosines = []
        for step in self._cosines:
            this_step = []
            for layer in step:
                if index:
                    this_step.append(layer[tuple(sorted(index))])
                else:
                    # sum all the values above the diagonal, sum all subtract the diagonal, 
                    # divide through by (sum(range(classes)), range starts at 0.
                    this_cosine = np.sum(layer)-np.sum(np.diagonal(layer))
                    # For some fucking reason I can't do this division on the same line as the eq above, python just ignores it idk?!?!?
                    this_cosine = this_cosine/sum(range(len(layer)))
                    this_step.append(this_cosine)
            cosines.append(this_step)
        return cosines
    
    def get_self_cosines(self, index = None):
        """ Returns cosine distance between datapoints of the same class.
        If an index is chosen, that will be the only class, otherwise classes will be averaged."""
        cosines = []
        for step in self._cosines:
            this_step = []
            for layer in step:
                if index:
                    this_step.append(np.diagonal(layer)[index])
                else: 
                    this_step.append(np.mean(np.diagonal(layer)))
            cosines.append(this_step)
        return cosines
    
    
                
        
                
            