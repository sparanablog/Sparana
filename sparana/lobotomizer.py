import numpy as np
import cupy as cp
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sparana.parameter_selection import get_k_biggest
from sparana.parameter_selection import get_k_smallest
from sparana.parameter_selection import get_normal_high
from sparana.model import model
from sparana.layers import full_relu_layer

def get_MAV_module(lobo, data):
    ''' This will run and store the mean activated values in the metric matrices in the class, sorts the list or whatever'''

    for i in data:
        if lobo._model._layer_type == 'Sparse':
            this_layer_inputs = i.transpose()
        if lobo._model._layer_type == 'Full':
            this_layer_inputs = i
        if lobo._model._comp_type == 'GPU':
            this_layer_inputs = cp.array(this_layer_inputs)
        output = None

        layer_count = 0
        for layer in lobo._model.layers:
            output = layer.activate(this_layer_inputs)
            this_layer_inputs = output
            if lobo._model._layer_type == 'Sparse':
                lobo._weight_stats[layer_count] += layer.activate_weights(this_layer_inputs)
            # Convert the activatedd full layers to sparse matrices.
            if lobo._model._layer_type == 'Full':
                lobo._weight_stats[layer_count] += csr_matrix(layer.activate_weights(this_layer_inputs))
            layer_count += 1
        if lobo._layer_type == 'Sparse':
            output = output.transpose()
    lobo._weight_stats = [coo_matrix(i) for i in lobo._weight_stats]
    for i in lobo._weight_stats:
        i.data = abs(i/len(data))
    return

def get_MAAV_module(lobo, data):
    ''' MAAV is mean absolutes activated values'''
    for layer in lobo._model.layers:
        layer._dropout = None
    for i in data:
        if lobo._model._layer_type == 'Sparse':
            this_layer_inputs = i.transpose()
        if lobo._model._layer_type == 'Full':
            this_layer_inputs = i
        if lobo._model._comp_type == 'GPU':
            this_layer_inputs = cp.array(this_layer_inputs)
        output = None

        layer_count = 0
        for layer in lobo._model.layers:
            if lobo._model._layer_type == 'Sparse':
                lobo._weight_stats[layer_count] += abs(layer.activate_weights(this_layer_inputs))
            # Convert the activatedd full layers to sparse matrices.
            if lobo._model._layer_type == 'Full':
                if lobo._lobo_type == 'lobotomizer':
                    lobo._weight_stats[layer_count] += abs(coo_matrix(cp.asnumpy(layer.activate_weights(this_layer_inputs))))
                if lobo._lobo_type == 'parameter_selector':
                    lobo._weight_stats[layer_count] += abs(cp.asnumpy(layer.activate_weights(this_layer_inputs)))
            #This is easier than trying to fix a reshape problem for an output I don't need to see
            if layer._activation_type != 'Softmax':
                output = layer.activate(this_layer_inputs)
            this_layer_inputs = output
            layer_count += 1
        if lobo._model._layer_type == 'Sparse':
            output = output.transpose()
    # Convert stuff here
    if lobo._lobo_type == 'lobotomizer':
        lobo._weight_stats = [coo_matrix(i) for i in lobo._weight_stats]
    for i in lobo._weight_stats:
        i = i/len(data)
    for layer in lobo._model.layers:
        layer._dropout = lobo._model._dropout
    return


def get_absolute_values_module(lobo):
    ''' Stores the sorted list or whatever, either of these will just replace what is already there'''
    if lobo._model._comp_type == 'GPU':
        lobo._weight_stats = [coo_matrix(abs(i.weights.get())) for i in lobo._model.layers]
    if lobo._model._comp_type == 'CPU':
        lobo._weight_stats = [coo_matrix(abs(i.weights)) for i in lobo._model.layers]
    return

class lobotomizer:
    
    ''' All stats arrays, sparse or no will be stored on the CPU ram, otherwise this will simply double the GPU memory requirements.
    These operations would be sped up on a GPU, but are run much less than training.'''
    
    def __init__(self, model):
        self._model = model
        self._lobo_type = 'lobotomizer'
        self._weight_stats = [coo_matrix(i._weights.shape) for i in self._model.layers]
        self._AV_datapoints = 0
                
    def get_MAV(self, data):
        ''' This will run and store the mean activated values in the metric matrices in the class, sorts the list or whatever'''
        get_MAV_module(self, data)        
        return
    
    def get_MAAV(self, data):
        ''' MAAV is mean absolutes activated values'''
        get_MAAV_module(self, data)
        return
    
    def MAAV_difference(self, main_data, target_data, print_nonzero_stats = False):
        ''' Doing this in a way that involves me rewriting the least code. Feed in both the main data and the target data. 
        This calculates the difference in MAAV values (target_data - main_data). The parameters with the largest value will be more active in the target class than in the main data. Remove some using prune_largest() '''
        # Get the MAAVs for the main data
        get_MAAV_module(self, main_data)
        # Move the data to a new array
        if print_nonzero_stats == True:
            print([i.nnz for i in self._weight_stats])
        main_data_MAAVs = self._weight_stats
        #reinitialize the weight stats sparse arrays
        self._weight_stats = [coo_matrix(i._weights.shape) for i in self._model.layers]
        # Get MAAVs for the target data
        get_MAAV_module(self, target_data)
        if print_nonzero_stats == True:
            print([i.nnz for i in self._weight_stats])
        # do the subtract
        for i in range(len(main_data_MAAVs)):
            self._weight_stats[i] = coo_matrix(self._weight_stats[i] - main_data_MAAVs[i])
        if print_nonzero_stats == True:
            print([i.nnz for i in self._weight_stats])
        return

    
    def get_absolute_values(self):
        ''' Stores the sorted list or whatever, either of these will just replace what is already there'''
        get_absolute_values_module(self)
        return
    
    def get_sparse_masks(self):
        """ Need to set the sparse training masks for selected training here"""
        for i in self._model._layers:
            i._sparse_training_mask = i._weights!=0
        return
    
    def get_random(self):
        """ Gets randomized stats matrices, so prune smallest prunes random weights"""
        
        return 
    
    def get_negative_values(self):
        if self._model._comp_type == 'GPU':
            self._weight_stats = [coo_matrix(i.weights.get()) for i in self._model.layers]
        if self._model._comp_type == 'CPU':
            self._weight_stats = [coo_matrix(i.weights) for i in self._model.layers]
        for i in range(len(self._model.layers)):
            self._weight_stats[i].data[self._weight_stats[i].data > 0] = 0
            self._weight_stats[i].eliminate_zeros()
            self._weight_stats[i].data = abs(self._weight_stats[i].data)
        
        return
    
    def get_positive_values(self):
        if self._model._comp_type == 'GPU':
            self._weight_stats = [coo_matrix(i.weights.get()) for i in self._model.layers]
        if self._model._comp_type == 'CPU':
            self._weight_stats = [coo_matrix(i.weights) for i in self._model.layers]
        for i in range(len(self._model.layers)):
            self._weight_stats[i].data[self._weight_stats[i].data < 0] = 0
            self._weight_stats[i].eliminate_zeros()
            self._weight_stats[i].data = abs(self._weight_stats[i].data)
        
        return
    
    def get_activation_ranks(self, data = None):
        """ Ranks the weights for each activation so that I can remove the smallest x% of weights from each activation, 
        not just the smallest weights from the whole weight matrix..................."""
        if data is not None:
            self._lobo_type = 'parameter_selector'
            self._weight_stats = [np.zeros(i._weights.shape) for i in self._model.layers]            
            get_MAAV_module(self, data)
            self._lobo_type = 'lobotomizer'
        else:
            if self._model._comp_type == 'GPU':
                self._weight_stats = [abs(i.weights.get()) for i in self._model.layers]
            if self._model._comp_type == 'CPU':
                self._weight_stats = [abs(i.weights) for i in self._model.layers]
        for i in range(len(self._weight_stats)):
            temp = []
            for j in self._weight_stats[i]:
                # This is surely not the most efficient way of doing this, there is a function
                # somewhere but I can't find it, so this will do.
                argsort = np.argsort(j)
                ranks = np.zeros(len(j))
                
                # Look at what the difference between the MAAV and absolute array structures, probably an indexing problem
                
                for k in range(len(j)):
                    ranks[argsort[k]] = k
                temp.append(ranks)
            self._weight_stats[i] = coo_matrix(np.array(temp))
        return
            
    
    def prune_smallest(self, prune_ratio = None, prune_number = None, print_stats = False, layers = None, prune_final_layer = False, zero_stats = True):
        ''' Prunes the weights in the model class.
        Using the smallest values from weight stats to prune.
        Sparse matrices will be reconstructed and assigned to the layer classes.
        Layers needs to be a list of ratios for eack layer to be pruned to. I can just not include the final layer.
        There are no checks or errors on here, so pay attention to the number of layers and the number of ratios input.'''
        
        # Sparse GPU weights need to be reassigned, dont support index based assignment, full GPU, and sparse, and full CPU 
        # can be assigned, I will need to run eliminate zeros.
        if layers:
            for i in range(len(layers)):
                
                if self._model._layer_type == 'Sparse' and self._model._comp_type == 'GPU':
                    # Copy weight matrix to CPU ram as a COO matrix
                    cpu_coo_matrix = self._model.layers[i]._weights.get().tocoo()
                    # Number of parameters to be removed                    
                    remove = int(layers[i]*cpu_coo_matrix.nnz)
                    if print_stats:
                        print('Pruning ', remove,' parameters from ', len(cpu_coo_matrix.data), ' parameters in layer ', i)
                    # List of indices of parameters to be removed
                    sortlist = np.argsort(self._layer_stats[i].data)[:remove]
                    # New COO matrix with parameters removed
                    cpu_coo_matrix = coo_matrix((cpu_coo_matrix.data[sortlist], (cpu_coo_matrix.row[sortlist], cpu_coo_matrix.col[sortlist])), shape = cpu_coo_matrix.shape)                                        
                    # Copy back to GPU in the layer class as the original CSR matrix
                    self._model.layers[i]._weights = cp.sparse.csr_matrix(cpu_coo_matrix)
                else:
                    if layers[i] != None:
                        # Number of parameters to be removed
                        if layers[i] <1:
                            remove = np.size(self._model.layers[i]._weights) *(layers[i] - (1-self._weight_stats[i].getnnz()/np.size(self._model.layers[i]._weights)))
                            remove = int(remove)
                        if layers[i]>1 or layers[i] == 0:
                            #
                            remove = layers[i]
                            
                        if print_stats:
                            print('Pruning ', remove,' parameters from ', self._weight_stats[i].nnz, ' parameters in layer ', i)
                        # List of indices of parameters to be removed
                        sortlist = np.argsort(self._weight_stats[i].data)[:remove]
                        # Loop through and set weights to 0
                        for j in sortlist:
                            
                            self._model.layers[i]._weights[self._weight_stats[i].row[j], self._weight_stats[i].col[j]] = 0
                            if zero_stats:
                                self._weight_stats[i].data[j] = 0
                        self._weight_stats[i].eliminate_zeros()
                            
        if not layers:
            # Not pruning the last layer, the model begins to fail quickly when this layer is pruned.
            if prune_final_layer == True:
                final_layer = 0
            if prune_final_layer == False:
                final_layer = 1
            for i in range(len(self._model.layers)-final_layer):
                
                if self._model._layer_type == 'Sparse' and self._model._comp_type == 'GPU':
                    # Copy weight matrix to CPU ram as a COO matrix
                    cpu_coo_matrix = self._model.layers[i]._weights.get().tocoo()
                    # Number of parameters to be removed
                    if prune_ratio:
                        remove = int(prune_ratio*cpu_coo_matrix.nnz)
                    if prune_number:
                        remove = prune_number
                    if print_stats:
                        print('Pruning ', remove,' parameters from ',cpu_coo_matrix.nnz, ' parameters in layer ', i)
                    # List of indices of parameters to be removed
                    sortlist = np.argsort(self._weight_stats[i].data)[:remove]
                    # New COO matrix with parameters removed
                    cpu_coo_matrix = coo_matrix((cpu_coo_matrix.data[sortlist], (cpu_coo_matrix.row[sortlist], cpu_coo_matrix.col[sortlist])), shape = cpu_coo_matrix.shape)                                        
                    # Copy back to GPU in the layer class as the original CSR matrix
                    self._model.layers[i]._weights = cp.sparse.csr_matrix(cpu_coo_matrix)
                else:
                    # Number of parameters to be removed
                    if prune_ratio:
                        remove = int(prune_ratio*self._weight_stats[i].getnnz())
                    if prune_number:
                        remove = prune_number
                    if print_stats:
                        print('Pruning ', remove,' parameters from ',' parameters in layer ', i)
                    # List of indices of parameters to be removed
                    sortlist = np.argsort(self._weight_stats[i].data)[:remove]
                    # Loop through and set weights to 0. There is probably a faster way to do this.
                    for j in sortlist:
                        self._model.layers[i]._weights[self._weight_stats[i].row[j], self._weight_stats[i].col[j]] = 0
            
        return
    
    def prune_biggest(self, prune_ratio = None, prune_number = None, print_stats = False, layers = None, prune_final_layer = False):
        '''Just going to set the weight_stats to negative and prune smallest'''
        for i in self._weight_stats:
            i.data = -i.data
        # then if I do this I will need to revert the stats in case I want to run this again with more
        for i in self._weight_stats:
            i.data = -i.data 
        return
    
    def prune_all_negative(self, layers = None, prune_ratio = None):
        """ Just prunes the weights of a matrix that are negative, I have not added the option of choosing what ratio to
        remove, but I might depending on how experiments go. """
        if layers:
            for i in range(len(layers)):
                if layers[i] == True:
                    self._model._layers[i]._weights[self._model._layers[i]._weights < 0] = 0
        else:
            for layer in self._model._layers:
                layer._weights[layer._weights < 0] = 0
        return

class vulcanizer:
    
    ''' This is for splitting a smaller model off the main model, which can then be trained in a memory/compute restricted system, and the parameters can be reinserted into the main model.'''
    
    def __init__(self, model, selection_type = 'max', std = None):
        self._model = model
        self._lobo_type = 'parameter_selector'
        if model._layer_type == 'Sparse':
            self._weight_stats = [coo_matrix(i._weights.shape) for i in self._model.layers]
        if model._layer_type == 'Full':
            self._weight_stats = [np.zeros(i._weights.shape) for i in self._model.layers]
        self._submodel = None
        self._coordinates = None
        self._average_zeros = None
        self._activation_selection = selection_type
        self._std = std
        self._mean_zeros = []
        
        
    def get_MAV(self, data):
        ''' This will run and store the mean activated values in the metric matrices in the class, sorts the list or whatever'''
        get_MAV_module(self, data)
        # Do a check before converting here
        for i in self._weight_stats:
            i = cp.asnumpy(i)
        return
    
    def get_MAAV(self, data):
        ''' MAAV is mean absolutes activated values'''
        get_MAAV_module(self, data)
        return

    
    def get_absolute_values(self):
        ''' The absolute values of the weights'''
        get_absolute_values_module(self)
        return
    
    def get_average_zeros(self, inputs):
        _ = self._model.outputs(input)
        for i in self._model.layers:
            self._mean_zeros.append(np.mean(i._outputs == 0))
        return
    
    def split_model(self, sizes, new_last_layer = False):
        ''' splits the model, returns a submodel, the sizes input is an array of the sizes of the final *x* layers.
        Define the last layer, it should be the same number of classes as the original. If there are 10 classes, the sizes array should look something like [50, 50, 10]. I dont have the new_last_layer bit built yet, this is a reminder, because I might want to.
        
        I am writing all of these operations explicitly and operating on new variables, this is for my own clarity.'''
        
        start = len(self._weight_stats) - len(sizes)        
        these_layers = []
        these_biases = []

        
        if len(self._mean_zeros) == 0:
            print ('You need to run get_average_zeros')
            return
       
        
        for i in range(len(sizes)):
            if self._model._comp_type == 'GPU':
                these_layers.append(cp.asnumpy(self._model.layers[start+i]._weights))
                biases.append(cp.asnumpy(self._model.layers[start+i]._biases))
            else:
                these_layers.append(self._model.layers[start+i]._weights)
                these_biases.append(self._model.layers[start+i]._biases)
        
        for i in range(len(sizes)-1):
                       
            # Get the indices of the max parameters/values
            if self._activation_selection == 'max':
                indices = get_max_columns(these_layers[i], sizes[i])
            if self._activation_selection == 'normal':
                indices = get_normal_columns(these_layers[i], sizes[i], self._std)
            #if self._activation_selection == 'uniform':
                #Just a RNG, could implement it here, don't need to sort anything. 
            
            self._coordinates.append(indices)
            
            # Indices of rows to be removed
            remove_these = np.delete(np.arange(these_layers[i].shape[1], indices))
                                     
            new_weights = np.delete(these_layers[i], remove_these, axis = 1)
                        
            # Get the means of the removed rows
            means = np.delete(these_layers[i], indices, axis = 1)
            means = np.mean(means, axis = i)
            
            # Append the means to the new weights
            new_weights = np.append(new_weights.T, [means*len(remove_these)*self._mean_zeros[i]], axis = 0).T
            
            # Put this array back into these_layers
            these_layers[i] = new_weights
            
            #Do all of the same operations with the axes flipped, to append you need to transpose twice.
            next_new_weights = np.delete(these_layers[i+1], remove_these, axis = 0)
            next_new_means = np.delete(these_layers[i+1], indices, axis = 0)
            next_new_means = np.mean(next_new_means, axis = 0)
            next_new_weights = np.append(next_new_weights, [next_new_means], axis = 0)
            these_layers[i+1] = next_new_weights
            
            # Now do the biases
            these_biases[i] = np.delete(these_biases[i], remove_these)
            
        
        # I need a new loop here to go through all of the layers, the last loop was not long enough
        for i in range(len(these_layers)):
            if mymodel.layers[i]._layer_type == 'Full':
                if mymodel.layers[i]._activation_type == 'Relu':
                    if mymodel.layers[i]._comp_type == 'CPU':
                        these_layers[i] = full_relu_layer(size = these_layers[i].shape[1], weights = these_layers[i], biases = these_biases[i])
                    if mymodel.layers[i]._comp_type == 'GPU':
                        these_layers[i] = full_relu_layer(size = these_layers[i].shape[1], weights = cp.array(these_layers[i]), biases = cp.array(these_biases[i]))
                if mymodel.layers[i]._activation_type == 'Linear':
                    if mymodel.layers[i]._comp_type == 'CPU':
                        these_layers[i] = full_linear_layer(size = these_layers[i].shape[1], weights = these_layers[i], biases = these_biases[i])
                    if mymodel.layers[i]._comp_type == 'GPU':
                        these_layers[i] = full_linear_layer(size = these_layers[i].shape[1], weights = cp.array(these_layers[i]), biases = cp.array(these_biases[i]))
                if mymodel.layers[i]._activation_type == 'Softmax':
                    if mymodel.layers[i]._comp_type == 'CPU':
                        these_layers[i] = full_softmax_layer(size = these_layers[i].shape[1], weights = these_layers[i], biases = these_biases[i])
                    if mymodel.layers[i]._comp_type == 'GPU':
                        these_layers[i] = full_Softmax_layer(size = these_layers[i].shape[1], weights = cp.array(these_layers[i]), biases = cp.array(these_biases[i]))    
            if mymodel.layers[i]._layer_type == 'Sparse':
                print('Sparse not implemented yet')
                break
        
        # Put the models in here
        self._submodel = model(input_size = None,
                         layers = these_layers,
                         comp_type = 'GPU')        

        return self._submodel
    
    def mind_meld(self):
        ''' returns the parameters into the original model'''
        return
    
    def revert_weights(self):
         ''' This is to return the consolidated weights to their original state after a training step''' 
        
         return
    
class parameter_selector:
    
    ''' All stats arrays, sparse or no will be stored on the CPU ram, otherwise this will simply double the GPU memory requirements.
    These operations would be sped up on a GPU, but are run much less than training.'''
    
    def __init__(self, model):
        self._model = model
        self._lobo_type = 'parameter_selector'
        if model._layer_type == 'Sparse':
            self._weight_stats = [coo_matrix(i._weights.shape) for i in self._model.layers]
        if model._layer_type == 'Full':
            self._weight_stats = [np.zeros(i._weights.shape) for i in self._model.layers]
        # TODO This is a dummy variable for comparing the stats of parameters when different data is passed through
        self._other_weight_stats = None
        # This is a list of ratios to keep track of which layers have had their parameters assessed.
        self._mask_ratios = np.zeros((len(self._weight_stats)))
        # Keeps track of whether the weights associated with the new class have been added
        self._class_weights_added = False
        # subsets assign random integers to the weight matrices to choose different subsets to train over
        self._subsets = None
                                     
                
    def get_MAV(self, data):
        ''' This will run and store the mean activated values in the metric matrices in the class, sorts the list or whatever'''
        get_MAV_module(self, data)
        # Do a check before converting here
        for i in self._weight_stats:
            i = cp.asnumpy(i)
        return
    
    def get_MAAV(self, data):
        ''' MAAV is mean absolutes activated values'''
        get_MAAV_module(self, data)
        return

    
    def get_absolute_values(self):
        ''' The absolute values of the weights'''
        get_absolute_values_module(self)
        return
    
    def get_top(self, ratio = None, number = None, layers = None, subset = None):
        ''' Returns the top k parameters, inputs should be either lists, or a single number.'''
        # Loop over the weight stats, get the top, store in the layer class and free up the space with weight_stat = None
        if not layers:
            for i in range(len(self._weight_stats)):
                # I have to convert this to a numpy array
                these_stats = self._weight_stats[i].toarray()
                indices = get_k_biggest([these_stats], ratio)
                # Initalize the bitmask
                self._model.layers[i]._sparse_training_mask = cp.zeros(self._model.layers[i]._weights.shape)
                # Keep track of the ratios
                self._mask_ratios[i] = ratio
                if not subset:
                    for j in indices[0]:
                        self._model.layers[i]._sparse_training_mask[j[0]][j[1]] = 1
                if subset:
                    for j in indices[0]:
                        if self._subsets[i][j[0]][j[1]] == subset:
                            self._model.layers[i]._sparse_training_mask[j[0]][j[1]] = 1
        if layers:
            for i in range(len(layers)):
                if layers[i] == 0:
                    self._model.layers[i]._sparse_training_mask = None
                if layers[i] != 0:
                    these_stats = self._weight_stats[i].toarray()
                    indices = get_k_biggest([these_stats], layers[i])
                    # Initialize the bitmask
                    self._model.layers[i]._sparse_training_mask = cp.zeros(self._model.layers[i]._weights.shape)
                    # Keep track of the ratios
                    self._mask_ratios[i] = ratio
                    if not subset:
                        for j in indices[0]:
                            self._model.layers[i]._sparse_training_mask[j[0]][j[1]] = 1
                    if subset:
                        for j in indices[0]:
                            if self._subsets[i][j[0]][j[1]] == subset:
                                self._model.layers[i]._sparse_training_mask[j[0]][j[1]] = 1
        return
    
    def get_bottom(self, ratio, number = None, layers = None, subset = None):
        ''' Returns the bottom k parameters, inputs should be either lists, or a single number.'''
        # Loop over the weight stats, get the top, store in the layer class and free up the space with weight_stat = None
        if not layers:
            for i in range(len(self._weight_stats)):
                these_stats = self._weight_stats[i].toarray()
                indices = get_k_smallest([these_stats], ratio)
                # Initalize the bitmask
                self._model.layers[i]._sparse_training_mask = cp.zeros(self._model.layers[i]._weights.shape)
                # Keep track of the ratios
                self._mask_ratios[i] = ratio
                if not subset:
                    for j in indices[0]:
                        self._model.layers[i]._sparse_training_mask[j[0]][j[1]] = 1
                if subset:
                    for j in indices[0]:
                        if self._subsets[i][j[0]][j[1]] == subset:
                            self._model.layers[i]._sparse_training_mask[j[0]][j[1]] = 1
        if layers:
            for i in range(len(layers)):
                if layers[i] == 0:
                    self._model.layers[i]._sparse_training_mask = None
                if layers[i] != 0:
                    these_stats = self._weight_stats[i].toarray()
                    indices = get_k_smallest([these_stats], ratio)
                    # Initialize the bitmask
                    self._model.layers[i]._sparse_training_mask = cp.zeros(self._model.layers[i]._weights.shape)
                    # Keep track of the ratios
                    self._mask_ratios[i] = ratio
                    if not subset:
                        for j in indices[0]:
                            self._model.layers[i]._sparse_training_mask[j[0]][j[1]] = 1
                    if subset:
                        for j in indices[0]:
                            if self._subsets[i][j[0]][j[1]] == subset:
                                self._model.layers[i]._sparse_training_mask[j[0]][j[1]] = 1
        return
    
    def add_output_class(self, output_class):
        '''This adds all of the weights that are associated with a new output class. '''
        self._model.layers[-1]._sparse_training_mask[:,output_class] = 1
        self._class_weights_added = True
        return
    
    def print_ratios(self):
        print('The ratios of parameters for each layer are:', self._mask_ratios)
        if self._class_weights_added:
            print('Output class weights have been added')
        else:
            print('Output class weights have NOT been added')
            
        return
    
    def get_top_gaussian(self, ratio = None, std = None, layers = None, subset = None):
        ''' Returns a selection of K parameters chosen from a gaussian distribution centred on the top values.'''
        # Loop over the weight stats, get the top, store in the layer class and free up the space with weight_stat = None
        if not layers:
            for i in range(len(self._weight_stats)):
                these_stats = self._weight_stats[i].toarray()
                indices = get_normal_high([these_stats], ratio, std)
                # Initalize the bitmask
                self._model.layers[i]._sparse_training_mask = cp.zeros(self._model.layers[i]._weights.shape)
                # Keep track of the ratios
                self._mask_ratios[i] = ratio
                if not subset:
                    for j in indices[0]:
                        self._model.layers[i]._sparse_training_mask[j[0]][j[1]] = 1
                if subset:
                    for j in indices[0]:
                        if self._subsets[i][j[0]][j[1]] == subset:
                            self._model.layers[i]._sparse_training_mask[j[0]][j[1]] = 1
        if layers:
            for i in range(len(layers)):
                if layers[i] == 0:
                    self._model.layers[i]._sparse_training_mask = None
                if layers[i] != 0:
                    these_stats = self._weight_stats[i].toarray()
                    indices = get_normal_high([these_stats], ratio, std)
                    # Initialize the bitmask
                    self._model.layers[i]._sparse_training_mask = cp.zeros(self._model.layers[i]._weights.shape)
                    # Keep track of the ratios
                    self._mask_ratios[i] = ratio
                    if not subset:
                        for j in indices[0]:
                            self._model.layers[i]._sparse_training_mask[j[0]][j[1]] = 1
                    if subset:
                        for j in indices[0]:
                            if self._subsets[i][j[0]][j[1]] == subset:
                                self._model.layers[i]._sparse_training_mask[j[0]][j[1]] = 1
        return
    
    def initialize_subsets(self, number_of_subsets):
        self._subsets = [np.floor(np.random.uniform(0, number_of_subsets, size = (i._weights.shape))) for i in self._model.layers]
        return
    
    
