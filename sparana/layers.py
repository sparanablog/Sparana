import numpy as np
import cupy as cp
from cupy.sparse import coo_matrix
from sparana.parameter_selection import get_normal_high
from sparana.numba_functions import sparse_coordinate_matmul

class full_relu_layer:
    
    def __init__(self, size = None, inputs = None, dropout = None, learning_rate = None, weights = None, biases = None):
        self._size = size
        self._layer_type = 'Full'
        self._activation_type = 'Relu'
        self._weights = weights
        self._biases = biases
        self._relu = None
        self._outputs = None
        self._comp_type = 'GPU'
        # Regularization parameters, and learning rates can be set for layers individually
        self._learning_rate = learning_rate
        self._dropout = dropout
        self._dropout_mask = None
        self._sparse_training_mask = None
        
        
    def layer_type(self):
        return self._layer_type
    
    def size(self):
        return self._size
    
    def activate_NG(self, inputs, ratio = None, distribution = None):
        '''Activate, NG for no gradient, needed to add too much to the regular activate module, was getting 
        convoluted. '''
               
        if distribution == 'binomial':
            if self._comp_type == 'GPU':
                self._dropout_mask = cp.random.binomial(1, ratio, size = self._weights.shape)
            if self._comp_type == 'CPU':
                self._dropout_mask = np.random.binomial(1, ratio, size = self._weights.shape)
        
        if self._comp_type == 'GPU':
            self._outputs = cp.dot(inputs, self._weights*self._dropout_mask)
            
        if self._comp_type == 'CPU':
            self._outputs = inputs@(self._weights*self._dropout_mask)
            
        self._outputs = self._outputs + self._biases
        self._relu = self._outputs>0
        self._outputs = self._outputs*self._relu
        return self._outputs
    
    def activate(self, inputs):
        if self._comp_type == 'GPU':
            self._outputs = cp.dot(inputs, self._weights)
        if self._comp_type == 'CPU':
            self._outputs = inputs@self._weights
        self._outputs = self._outputs + self._biases
        self._relu = self._outputs>0
        self._outputs = self._outputs*self._relu
        # Apply dropout to the outputs of this layer
        if self._dropout:
            if self._comp_type == 'CPU':
                self._dropout_mask = np.random.binomial(1, 1-self._dropout, size = self._outputs.shape)
            if self._comp_type == 'GPU':
                self._dropout_mask = cp.random.binomial(1, 1-self._dropout, size = self._outputs.shape)
            self._outputs = self._outputs*self._dropout_mask
        return self._outputs
    
    def activate_weights(self, inputs):
        if self._comp_type == 'GPU':
            return cp.multiply(self._weights, inputs[: , np.newaxis])
        if self._comp_type == 'CPU':
            return np.multiply(self._weights, inputs[: , cp.newaxis])
        
    
    @property
    def weights(self):
        return self._weights
    
    def scale_weights(self, scaling_factor):
        self._weights *= scaling_factor
        return
    
    @property
    def biases(self):
        return self._biases
    
    def get_gradients(self, layer_inputs, layer_error):
        ''' Returns an array for weights, and biases, and one for the previous layer'''
        if self._dropout:
            layer_error = layer_error*self._dropout_mask
        if self._comp_type == 'CPU':
            layer_error = layer_error*self._relu
            bias_gradients = np.sum(layer_error, axis = 0)
            weight_gradients = layer_inputs.transpose()@(layer_error)
            previous_layer_error = layer_error@self._weights.transpose()
        if self._comp_type == 'GPU':
            layer_error = layer_error*self._relu
            bias_gradients = cp.sum(layer_error, axis = 0)
            weight_gradients = cp.dot(layer_inputs.transpose(), layer_error)
            previous_layer_error = cp.dot(layer_error, self._weights.transpose())    
        return weight_gradients, bias_gradients, previous_layer_error
    
    def get_selected_gradients(self, layer_inputs, layer_error, parameters):
        ''' Returns an array of the gradients of the selected parameters for weights, and biases, and one for the previous layer'''
        
        
        # Do the thing above with sparse_parameter_matmul(x,y,parameters)
        
        
        return 
    
    def convert_comp_type(self):
        if self._comp_type == 'GPU':
            self._comp_type = 'CPU'
            self._weights = cp.asnumpy(self._weights)
            self._biases = cp.asnumpy(self._biases)
        
        if self._comp_type == 'CPU':
            self._comp_type = 'GPU'
            self._weights = cp.array(self._weights)
            self._biases = cp.array(self._biases)
                
class full_linear_layer:
    
    def __init__(self, size = None, inputs = None, dropout = None, learning_rate = None):
        self._size = size
        self._layer_type = 'Full'
        self._activation_type = 'Linear'
        self._weights = None
        self._biases = None
        self._relu = 1
        self._outputs = None
        self._comp_type = 'CPU'
        # Regularization parameters, and learning rates can be set for layers individually
        self._learning_rate = learning_rate
        self._dropout = dropout
        if self._comp_type == 'CPU' and dropout:
            self._dropout_mask = np.random.binomial(1, 1-self._dropout, size = self._weights.shape)
        if self._comp_type == 'GPU' and dropout:
            self._dropout_mask = cp.random.binomial(1, 1-self._dropout, size = self._weights.shape)
        self._sparse_training_mask = None
            
    def layer_type(self):
        return self._layer_type
    
    def size(self):
        return self._size
    
    def activate(self, inputs):
        if self._comp_type == 'GPU':
            if self._dropout:
                # Dropout masks are reset with every forward pass to be reused for calculating gradients.
                self._dropout_mask = cp.random.binomial(1, 1-self._dropout, size = self._weights.shape)
                self._outputs = cp.dot(inputs, self._weights*self._dropout_mask)
            else:
                self._outputs = cp.dot(inputs, self._weights)
        if self._comp_type == 'CPU':
            if self._dropout:
                self._dropout_mask = np.random.binomial(1, 1-self._dropout, size = self._weights.shape)
                self._outputs = inputs@(self._weights*self._dropout_mask)
            else:
                self._outputs = inputs@self._weights
        self._outputs = self._outputs + self._biases
        return self._outputs
    
    def activate_weights(self, inputs):
        if self._comp_type == 'GPU':
            return cp.multiply(self._weights, inputs[: , np.newaxis])
        if self._comp_type == 'CPU':
            return np.multiply(self._weights, inputs[: , cp.newaxis])
        
    def activate_NG(self, inputs, ratio = None, distribution = None):
        '''Activate, NG for no gradient, needed to add too much to the regular activate module, was getting 
        convoluted. '''
               
        if distribution == 'binomial':
            if self._comp_type == 'GPU':
                self._dropout_mask = cp.random.binomial(1, ratio, size = self._weights.shape)
            if self._comp_type == 'CPU':
                self._dropout_mask = np.random.binomial(1, ratio, size = self._weights.shape)
        
        if self._comp_type == 'GPU':
            self._outputs = cp.dot(inputs, self._weights*self._dropout_mask)
            
        if self._comp_type == 'CPU':
            self._outputs = inputs@(self._weights*self._dropout_mask)
        
        self._outputs = self._outputs + self._biases
        return self._outputs
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def biases(self):
        return self._biases
    
    def get_gradients(self, layer_inputs, layer_error):
        ''' Returns an array for weights, and biases, and one for the previous layer'''
        if self._comp_type == 'CPU':
            bias_gradients = np.sum(layer_error, axis = 0)
            weight_gradients = layer_inputs.transpose()@(layer_error)
            if self._dropout:
                previous_layer_error = layer_error@(self._dropout_mask*self._weights).transpose()
            else:
                previous_layer_error = layer_error@self._weights.transpose()
        if self._comp_type == 'GPU':
            bias_gradients = cp.sum(layer_error, axis = 0)
            weight_gradients = cp.dot(layer_inputs.transpose(), layer_error)
            if self._dropout:
                previous_layer_error = cp.dot(layer_error, (self._dropout_mask*self._weights).transpose())
            else:
                previous_layer_error = cp.dot(layer_error, self._weights.transpose())
        
        return weight_gradients, bias_gradients, previous_layer_error
        
    def convert_comp_type(self):
        if self._comp_type == 'GPU':
            self._comp_type = 'CPU'
            self._weights = cp.asnumpy(self._weights)
            self._biases = cp.asnumpy(self._biases)
        
        if self._comp_type == 'CPU':
            self._comp_type = 'GPU'
            self._weights = cp.array(self._weights)
            self._biases = cp.array(self._biases)    

class full_softmax_layer:
    
    def __init__(self, size = None, inputs = None, dropout = None, learning_rate = None):
        self._size = size
        self._layer_type = 'Full'
        self._activation_type = 'Softmax'
        self._weights = None
        self._biases = None
        self._relu = 1
        self._outputs = None
        self._comp_type = 'CPU'
        # Regularization parameters, and learning rates can be set for layers individually
        self._learning_rate = learning_rate
        self._dropout = dropout
        if self._comp_type == 'CPU' and dropout:
            self._dropout_mask = np.random.binomial(1, 1-self._dropout, size = self._weights.shape)
        if self._comp_type == 'GPU' and dropout:
            self._dropout_mask = cp.random.binomial(1, 1-self._dropout, size = self._weights.shape)
        self._sparse_training_mask = None
        self._pre_softmax_values = None
        
            
    def layer_type(self):
        return self._layer_type
    
    def size(self):
        return self._size
    
    def activate(self, inputs):
        if self._comp_type == 'GPU':
            if self._dropout:
                # Dropout masks are reset with every forward pass to be reused for calculating gradients.
                self._dropout_mask = cp.random.binomial(1, 1-self._dropout, size = self._weights.shape)
                self._outputs = cp.dot(inputs, self._weights*self._dropout_mask)
            else:
                self._outputs = cp.dot(inputs, self._weights)
        if self._comp_type == 'CPU':
            if self._dropout:
                self._dropout_mask = np.random.binomial(1, 1-self._dropout, size = self._weights.shape)
                self._outputs = inputs@(self._weights*self._dropout_mask)
            else:
                self._outputs = inputs@self._weights
        self._outputs = self._outputs + self._biases
        self._pre_softmax_values = self._outputs
        self._outputs = np.exp(self._outputs)
        self._outputs = self._outputs/(np.sum(self._outputs, axis = 1)).reshape(len(self._outputs), 1)
        return self._outputs
    
    def activate_weights(self, inputs):
        if self._comp_type == 'GPU':
            return cp.multiply(self._weights, inputs[: , np.newaxis])
        if self._comp_type == 'CPU':
            return np.multiply(self._weights, inputs[: , cp.newaxis])
        
    def activate_NG(self, inputs, ratio = None, distribution = None):
        '''Activate, NG for no gradient, needed to add too much to the regular activate module, was getting 
        convoluted. '''
               
        if distribution == 'binomial':
            if self._comp_type == 'GPU':
                self._dropout_mask = cp.random.binomial(1, ratio, size = self._weights.shape)
            if self._comp_type == 'CPU':
                self._dropout_mask = np.random.binomial(1, ratio, size = self._weights.shape)
        
        if self._comp_type == 'GPU':
            self._outputs = cp.dot(inputs, self._weights*self._dropout_mask)
            
        if self._comp_type == 'CPU':
            self._outputs = inputs@(self._weights*self._dropout_mask)
        
        self._outputs = self._outputs + self._biases
        return self._outputs
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def biases(self):
        return self._biases
    
    def get_gradients(self, layer_inputs, layer_error):
        ''' Returns an array for weights, and biases, and one for the previous layer'''
        
        if self._dropout:
            layer_error = layer_error*self._dropout_mask
        
        if self._comp_type == 'CPU':
            bias_gradients = np.sum(layer_error, axis = 0)
            weight_gradients = layer_inputs.transpose()@(layer_error)
            if self._dropout:
                previous_layer_error = layer_error@(self._dropout_mask*self._weights).transpose()
            else:
                previous_layer_error = layer_error@self._weights.transpose()
        if self._comp_type == 'GPU':
            bias_gradients = cp.sum(layer_error, axis = 0)
            weight_gradients = cp.dot(layer_inputs.transpose(), layer_error)
            if self._dropout:
                previous_layer_error = cp.dot(layer_error, (self._dropout_mask*self._weights).transpose())
            else:
                previous_layer_error = cp.dot(layer_error, self._weights.transpose())
            
        return weight_gradients, bias_gradients, previous_layer_error
        
    def convert_comp_type(self):
        if self._comp_type == 'GPU':
            self._comp_type = 'CPU'
            self._weights = cp.asnumpy(self._weights)
            self._biases = cp.asnumpy(self._biases)
        
        if self._comp_type == 'CPU':
            self._comp_type = 'GPU'
            self._weights = cp.array(self._weights)
            self._biases = cp.array(self._biases)    

            
class sparse_relu_layer:
    
    def __init__(self, size, weights = None, biases = None, inputs = None, dropout = None, learning_rate = None):
        self._size = size
        self._layer_type = 'Sparse'
        self._activation_type = 'Relu'
        self._weights = weights
        self._biases = biases
        self._relu = None
        self._outputs = None
        # Default to running on GPU, if the sparse model isn't going to fit in GPU memory, you were fucked anyway.
        self._comp_type = 'GPU'
        # Regularization parameters, and learning rates can be set for layers individually
        self._learning_rate = learning_rate
        self._dropout = dropout
        self._weight_gradients = None        
    
    @property    
    def get_inputs(self):
        return self._inputs
       
    def activate(self, inputs):
        if self._comp_type == 'GPU':
            self._outputs = self._weights.dot(inputs)
        if self._comp_type == 'CPU':
            # use the @ operator
            self._outputs = inputs@self._weights
        self._outputs = self._outputs + self._biases[: , np.newaxis]
        self._relu = self._outputs>0
        self._outputs = self._outputs*self._relu
        return self._outputs

    @property
    def softmax_activate(self):
        dot_product = self._inputs@self._weights
        add_biases = dot_product + self._biases
        softmax = np.array([[np.exp(i)/sum([np.exp(j) for j in k]) for i in k] for k in add_biases])
        return softmax
        
    @property
    def weights(self):
        return self._weights
    
    def activate_weights(self, inputs):
        act_weights = self._weights.multiply(np.transpose(inputs))
        return act_weights
    
    @property
    def biases(self):
        return self._biases
    
    def get_coordinates(self):
        self._rows = self._weights.tocoo().transpose().row
        self._columns = self._weights.tocoo().transpose().col
    
    def get_gradients(self, layer_inputs, layer_error, coordinates):
        #if self._weight_gradients is None:
        self._weight_gradients = cp.zeros(self._weights.nnz)
        grads_shape = self._weights.shape
        layer_error = layer_error*(self._relu.transpose())
        bias_gradients = cp.sum(layer_error, axis = 0)
        previous_layer_error = self._weights.transpose().dot(layer_error.transpose()).transpose()
        
        tpb = 512
        bpg_x = int(coordinates.shape[0] /tpb)
        
        #sparse_coordinate_matmul[tpb, 200](layer_inputs, layer_error, self._weight_gradients, coordinates)
        
        #weight_gradients = sum(layer_inputs[self._rows,:].transpose()*layer_error[:,self._columns])
        weight_gradients = cp.dot(layer_inputs, layer_error)
        
        return weight_gradients, bias_gradients, previous_layer_error

class sparse_linear_layer:
    
    def __init__(self, size, weights = None, biases = None, inputs = None, dropout = None, learning_rate = None):
        self._size = size
        self._layer_type = 'Sparse'
        self._activation_type = 'Linear'
        self._weights = weights
        self._biases = biases
        self._relu = 1
        self._outputs = None
        # Default to running on GPU, if the sparse model isn't going to fit in GPU memory, you were fucked anyway.
        self._comp_type = 'GPU'
        # Regularization parameters, and learning rates can be set for layers individually
        self._learning_rate = learning_rate
        self._dropout = dropout
        self._rows = None
        self._columns = None
        self._weight_gradients = None
            
    @property    
    def get_inputs(self):
        return self._inputs
       
    def activate(self, inputs):
        if self._comp_type == 'GPU':
            self._outputs = self._weights.dot(inputs)
        if self._comp_type == 'CPU':
            # use the @ operator
            self._outputs = inputs@self._weights
        self._outputs = self._outputs + self._biases[: , np.newaxis]
        return self._outputs

    @property
    def softmax_activate(self):
        self._outputs = self._inputs@self._weights
        self._outputs = self._outputs + self._biases
        softmax = np.array([[np.exp(i)/sum([np.exp(j) for j in k]) for i in k] for k in self._outputs])
        return softmax
        
    @property
    def weights(self):
        return self._weights
    
    @property
    def activate_weights(self):
        act_weights = self._weights.multiply((np.transpose(self._inputs)))
        return act_weights
    
    @property
    def biases(self):
        return self._biases
    
    def get_coordinates(self):
        self._rows = self._weights.tocoo().transpose().row
        self._columns = self._weights.tocoo().transpose().col
               
    def get_gradients(self, layer_inputs, layer_error, coordinates):
        if self._weight_gradients is None:
            self._weight_gradients = cp.zeros(self._weights.nnz)

        grads_shape = self._weights.shape
        bias_gradients = cp.sum(layer_error, axis = 0)
        
        previous_layer_error = self._weights.transpose().dot(layer_error.transpose()).transpose()
       
        tpb = 512
        bpg_x = int(coordinates.shape[0] /tpb)
        
        #sparse_coordinate_matmul[tpb, 200](layer_inputs, layer_error, self._weight_gradients, coordinates)
        
        #weight_gradients = sum(layer_inputs[self._rows,:].transpose()*layer_error[:,self._columns])
        weight_gradients = cp.dot(layer_inputs, layer_error)
        
        return weight_gradients, bias_gradients, previous_layer_error

        
      