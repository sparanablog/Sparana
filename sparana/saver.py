import numpy as np
import cupy as cp
import pickle
from sparana.layers import full_linear_layer
from sparana.layers import full_relu_layer
from sparana.layers import full_softmax_layer
from cupy.sparse import coo_matrix
from cupy.sparse import csr_matrix

class model_saver:
    
    def __init__(self, model):
        self._model = model
        if self._model._layer_type == 'Sparse':
            if self._model._comp_type == 'GPU':
                self._model_arrays = [(i._weights.get(), i._biases.get()) for i in self._model._layers] 
            if self._model._comp_type == 'CPU':
                self._model_arrays = [(i._weights.copy(), np.array(i._biases)) for i in self._model._layers]
        if self._model._layer_type == 'Full':
            if self._model._comp_type == 'GPU':
                self._model_arrays = [(i._weights.get(), i._biases.get()) for i in self._model._layers]
            if self._model._comp_type == 'CPU':
                self._model_arrays = [(np.array(i._weights), np.array(i._biases)) for i in self._model._layers]
        self._sparse_parameters = None
            
        
        
    def store_model(self):
        ''' Stores the current state of the model. '''
        if self._model._layer_type == 'Sparse':
            if self._model._comp_type == 'GPU':
                self._model_arrays = [(i._weights.get(), np.array(i._biases)) for i in self._model._layers] 
            if self._model._comp_type == 'CPU':
                self._model_arrays = [(i._weights.copy(), np.array(i._biases)) for i in self._model._layers]
        if self._model._layer_type == 'Full':
            if self._model._comp_type == 'GPU':
                self._model_arrays = [(i._weights.get(), i._biases.get()) for i in self._model._layers]
            if self._model._comp_type == 'CPU':
                self._model_arrays = [(np.array(i._weights), np.array(i._biases)) for i in self._model._layers]
        return
    
    
    def restore_model(self):
        ''' Restores the weights stored in the model saver. '''
        if self._model._layer_type == 'Sparse':
            if self._model._comp_type == 'CPU':
                for i in range(self._model._depth):
                    self._model._layers[i]._weights = self._model_arrays[i][0].copy()
                    self._model._layers[i]._biases = np.array(self._model_arrays[i][1])
            if self._model._comp_type == 'GPU':
                for i in range(self._model._depth):
                    self._model._layers[i]._weights = cp.sparse.csr_matrix(self._model_arrays[i][0])
                    self._model._layers[i]._biases = cp.array(self._model_arrays[i][1])
        if self._model._layer_type == 'Full':
            if self._model._comp_type == 'GPU':
                for i in range(self._model._depth):
                        self._model._layers[i]._weights = cp.array(self._model_arrays[i][0])
                        self._model._layers[i]._biases = cp.array(self._model_arrays[i][1])
            if self._model._comp_type == 'CPU':
                for i in range(self._model._depth):
                        self._model._layers[i]._weights = np.array(self._model_arrays[i][0])
                        self._model._layers[i]._biases = np.array(self._model_arrays[i][1])
        return
    
    def restore_full_model(self):
        '''This is for restoring a model of full layers if the model has been converted to a sparse layers'''
        if self._model._comp_type == 'GPU':
            for i in range(self._model._depth):
                if self._model._layers[i]._activation_type == 'Linear':
                    self._model._layers[i] = full_linear_layer(size = self._model._layers[i]._size)
                if self._model._layers[i]._activation_type == 'Relu':
                    self._model._layers[i] = full_relu_layer(size = self._model._layers[i]._size)
                if self._model._layers[i]._activation_type == 'Softmax':
                    self._model._layers[i] = full_softmax_layer(size = self._model._layers[i]._size)
                self._model._layers[i]._weights = cp.array(self._model_arrays[i][0])
                self._model._layers[i]._biases = cp.array(self._model_arrays[i][1])
        if self._model._comp_type == 'CPU':
            for i in range(self._model._depth):
                if self._model._layers[i]._activation_type == 'Linear':
                    self._model._layers[i] = full_linear_layer(size = self._model._layers[i]._size)
                if self._model._layers[i]._activation_type == 'Relu':
                    self._model._layers[i] = full_relu_layer(size = self._model._layers[i]._size)
                if self._model._layers[i]._activation_type == 'Softmax':
                    self._model._layers[i] = full_softmax_layer(size = self._model._layers[i]._size)
                self._model._layers[i]._weights = np.array(self._model_arrays[i][0])
                self._model._layers[i]._biases = np.array(self._model_arrays[i][1])
        self._model._layer_type = 'Full'
        return

        
    def pickle_model(self, filename):
        ''' Stores the model in a pickle file. '''
        pickle.dump(self._model, open(filename, 'wb'))
        print('Model pickled')
        return
    
    def load_model(self, filename):
        ''' Loads the model from a pickle file. '''
        filelist = pickle.load(open(filename, 'rb'))
        if self._model._layer_type == 'Sparse':
            self._model_arrays = [(i[0].copy(), np.array(i[1])) for i in filelist]                
        if self._model._layer_type == 'Full':
            for i in range(self._model._depth):
                self._model.layers[i]._weights = filelist.layers[i]._weights
                self._model.layers[i]._biases = filelist.layers[i]._biases
            
        # Do a check that the layer type matches the weight datatype
    
    def transfer_partial_model(self, newmodel, output_classes = None, bias_constant = 0.1):
        ''' Transfers the weights stored in the saver into a new model, and initializes the final layer of the new model.'''
        # Put the weights and biases into the new model, put the size in there too so I can print the output too
        if newmodel._comp_type == 'GPU':
            for i in range(newmodel._depth - 1):
                newmodel._layers[i]._weights = cp.array(self._model_arrays[i][0])
                newmodel._layers[i]._biases = cp.array(self._model_arrays[i][1])
                newmodel._layers[i]._size = len(self._model_arrays[i][1])
        if self._model._comp_type == 'CPU':
            for i in range(self._model._depth - 1):
                newmodel._layers[i]._weights = np.array(self._model_arrays[i][0])
                newmodel._layers[i]._biases = np.array(self._model_arrays[i][1])
                newmodel._layers[i]._size = len(self._model_arrays[i][1])
        # Initialize a new final layer. 
        if output_classes == None:
            output_classes = newmodel._layers[-1]._size
        shape = (newmodel._layers[-2]._size, output_classes)
        if newmodel._comp_type == 'GPU':
            newmodel.layers[-1]._weights = cp.random.normal(0, np.sqrt(3. / sum(shape)), shape, dtype = np.float32)
            newmodel.layers[-1]._biases = cp.full(newmodel._layers[-1]._size, bias_constant, dtype = np.float32)
        if newmodel._comp_type == 'CPU':
            newmodel.layers[-1]._weights = np.random.normal(0, np.sqrt(3. / sum(shape)))
            newmodel.layers[-1]._biases = np.full(newmodel._layers[-1]._size, bias_constant)
     
    def load_sparse_parameters(self, filename):
        ''' Loads sparse parameters into the loader class, and into the model. 
        (I can't think of a real use for loading the parameters into the loader, and model seperately)'''
        parameters = pickle.load(open(filename, 'rb'))
        for i in range(len(parameters)):
            if parameters[i] == None:
                self._model._layers[i]._sparse_training_mask = None
            else:
            # Put the training masks in the layer objects, TODO turn this into a [0,1] mask
                # Put the individual weights in the weight matrices
                for j in range(parameters[i].nnz):
                    self._model._layers[i]._weights[parameters[i].row[j]][parameters[i].col[j]] = parameters[i].data[j]
                parameters[i].data = np.ones(len(parameters[i].data))
                self._model._layers[i]._sparse_training_mask = parameters[i].toarray()
        print('Inserted weights from ', filename, ' into the weight matrices')
        return
        
    def store_sparse_parameters(self):
        ''' This returns the parameters that can be stored in memory in the notebook, use pickle_sparse_parameters after this'''
        # What format will this give me, I need sparse matrices.
        parameters = []
        for i in self._model._layers:
            if i._sparse_training_mask is None:
                parameters.append(None)
            else:
                these_parameters = np.multiply(i._weights, i._sparse_training_mask)
                # Sparsify these_parameters
                these_parameters = csr_matrix(these_parameters, dtype = np.float32)
                these_parameters = these_parameters.tocoo()
                parameters.append((these_parameters, i._biases))
        self._sparse_parameters = parameters
        return 
    
    def pickle_sparse_parameters(self, filename):
        ''' Stores the sparse parameters in a pickle file. '''
        if self._sparse_parameters == None:
            print('No parameters stored')
            return
        pickle.dump(self._sparse_parameters, open(filename, 'wb'))
        print('Model pickled')
        return
    
    def restore_sparse_parameters(self):
        ''' Need a more specific name than sparse parameters. this will take some learning, drop the weights in'''
        if self._sparse_parameters == None:
            print('No parameters stored')
            return
        for i in range(len(self._sparse_parameters)):
            # Put the training masks in the layer objects, TODO turn this into a [0,1] mask
            #self._model._sparse_training_mask = None #parameters[i]
            # Put the individual weights in the weight matrices
            
            for j in range(self._sparse_parameters[i][0].nnz):
                self._model._layers[i]._weights[int(self._sparse_parameters[i][0].row[j])][int(self._sparse_parameters[i][0].col[j])] = self._sparse_parameters[i][0].data[j]
            
            # Replace full arrays, test
            #self._model._layers[i]._weights = np.multiply(self._model._layers[i]._weights, (self._sparse_parameters[i][0] == 0))
            #self._model._layers[i]._weights = self._model._layers[i]._weights + self._sparse_parameters[i][0]
            self._model._layers[i]._biases = self._sparse_parameters[i][1]

            
        print('Sparse parameters restored')
        return