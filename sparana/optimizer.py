import numpy as np
import cupy as cp
from sparana.parameter_selection import get_k_max

def get_gradients(opt, inputs, labels, coordinates = None, backward_layers = None, loss_function = 'MSE'):
    
    outputs = opt._model.outputs(inputs)
    # This is hard coded quadratic error. The error for the softmax is built in here for reasons.
    gradients = []
    if loss_function == 'MSE':
        if opt._model._comp_type == 'CPU':
            error = -(outputs - labels)
        if opt._model._comp_type == 'GPU':
            error = -(outputs - cp.array(labels))
    if loss_function == 'CE':
        if opt._model._comp_type == 'CPU':
            error = -(outputs - labels)
        if opt._model._comp_type == 'GPU':
            error = -(outputs - cp.array(labels))
        
    if backward_layers == None:
        backward_layers = len(opt._model.layers)
    
    for i in range(backward_layers):
        
        # All the layers after the first layer, gradients are calculated the same way.
        
        if i < len(opt._model.layers)-1:
            # For the last layer, feed in the error calculated from the outputs, for the middle layers
            # feed in the error from the following layer, and outputs from the previous layer
                
            if opt._model._layer_type == 'Full':
                weight_gradients, bias_gradients, error = opt._model.layers[-1-i].get_gradients(opt._model.layers[-2-i]._outputs, error)
            if opt._model._layer_type == 'Sparse':
                if opt._model.layers[-1-i]._activation_type == 'Relu':
                    weight_gradients, bias_gradients, error = opt._model.layers[-1-i].get_gradients(opt._model.layers[-2-i]._outputs, error*(opt._model.layers[-1-i]._relu.transpose()), coordinates[-1-i])
                if opt._model.layers[-1-i]._activation_type == 'Linear':
                    weight_gradients, bias_gradients, error = opt._model.layers[-1-i].get_gradients(opt._model.layers[-2-i]._outputs, error*opt._model.layers[-1-i]._relu, coordinates[-1-i])
                
            gradients.append((weight_gradients, bias_gradients))
        
        # For the first layer, need the inputs instead of the previous layers outputs.
        
        if i == len(opt._model.layers)-1:
            if opt._model._comp_type == 'CPU':
                weight_gradients, bias_gradients, error = opt._model.layers[-1-i].get_gradients(inputs, error*opt._model.layers[-1-i]._relu)
            if opt._model._comp_type == 'GPU':
                if opt._model._layer_type == 'Full':
                    weight_gradients, bias_gradients, error = opt._model.layers[-1-i].get_gradients(cp.array(inputs), error) #*opt._model.layers[-1-i]._relu)
                if opt._model._layer_type == 'Sparse':
                    weight_gradients, bias_gradients, error = opt._model.layers[-1-i].get_gradients(cp.array(inputs.transpose()), error*(opt._model.layers[-1-i]._relu.transpose()), coordinates[-1-i])
            gradients.append((weight_gradients, bias_gradients))
        
    # Gradients are appended in reverse order, reverse this to simplify applying training step
    gradients.reverse()
        
    return gradients

def selected_gradients(opt, inputs, labels, layers):
    
    outputs = opt._model.outputs(inputs)
    # This is hard coded quadratic error.
    gradients = []
    if opt._model._comp_type == 'CPU':
        error = -(outputs - labels)
    if opt._model._comp_type == 'GPU':
        error = -(outputs - cp.array(labels))
       
    ### This bit goes for i in layers: Got to start last and move backwards, 
        
    for i in range(len(opt._model.layers)):
            
        if i < len(opt._model.layers)-1:
            # For the last layer, feed in the error calculated from the outputs, for the middle layers
            # feed in the error from the following layer, and outputs from the previous layer
                
            if opt._model._layer_type == 'Full':
                weight_gradients, bias_gradients, error = opt._model.layers[-1-i].get_gradients(opt._model.layers[-2-i]._outputs, error*opt._model.layers[-1-i]._relu)
            if opt._model._layer_type == 'Sparse':
                if opt._model.layers[-1-i]._activation_type == 'Relu':
                    weight_gradients, bias_gradients, error = opt._model.layers[-1-i].get_gradients(opt._model.layers[-2-i]._outputs, error*(opt._model.layers[-1-i]._relu.transpose()))
                if opt._model.layers[-1-i]._activation_type == 'Linear':
                    weight_gradients, bias_gradients, error = opt._model.layers[-1-i].get_gradients(opt._model.layers[-2-i]._outputs, error)
                
            gradients.append((weight_gradients, bias_gradients))
        if i == len(opt._model.layers)-1:
            # For the first layer, feed in the error from the following layer, and the inputs
            if opt._model._comp_type == 'CPU':
                weight_gradients, bias_gradients, error = opt._model.layers[-1-i].get_gradients(inputs, error*opt._model.layers[-1-i]._relu)
            if opt._model._comp_type == 'GPU':
                if opt._model._layer_type == 'Full':
                    weight_gradients, bias_gradients, error = opt._model.layers[-1-i].get_gradients(cp.array(inputs), error*opt._model.layers[-1-i]._relu)
                if opt._model._layer_type == 'Sparse':
                    weight_gradients, bias_gradients, error = opt._model.layers[-1-i].get_gradients(cp.array(inputs.transpose()), error*(opt._model.layers[-1-i]._relu.transpose()))
            gradients.append((weight_gradients, bias_gradients))
        
    # Gradients are appended in reverse order, reverse thisto simplify applying training step
    gradients.reverse()
        
    return gradients

def filter_gradients(gradients, ratio):
    minimum = ratio*gradients.size
    mask = cp.reshape(cp.argsort(cp.abs(gradients), axis = None), gradients.shape)>minimum
    return cp.multiply(gradients, mask)

class sgd_optimizer:
    
    """ First attempt at building an optimizer, only uses quadratic cost function"""
    def __init__(self, model, learning_rate, l1_constant = None, l2_constant = None):
        self._model = model
        #if self._model.layers[0]._learning_rate != None:
        #    self._layer_learning_rates = True
        self._learning_rate = learning_rate
        self._gradients = []
        self._l2_constant = l2_constant
        self._l1_constant = l1_constant
        if self._model._layer_type == 'Sparse':
            self._sparse_coords = []
            for layer in self._model.layers:
                temp = layer._weights.tocoo()
                self._sparse_coords.append(cp.array([(i, j) for i, j in zip(cp.asnumpy(temp.row), cp.asnumpy(temp.col))]))
                temp = None
    
    def get_gradients(self, inputs, labels):
        if self._model._layer_type == 'Full':
            grads = get_gradients(self, inputs, labels)
        if self._model._layer_type == 'Sparse':
            grads = get_gradients(self, inputs, labels, self._sparse_coords)
        return grads
    
    def train_step(self, inputs, labels, filter_minimum = None):
        if self._model._layer_type == 'Full':
            grads = get_gradients(self, inputs, labels)
        if self._model._layer_type == 'Sparse':
            grads = get_gradients(self, inputs, labels, self._sparse_coords)
        for i in range(len(grads)):
            
            if self._model._layer_type == 'Full':
                #HMMM now I have to fuck around with shapes and stuff. 
                if filter_minimum:
                    # I need to replace the whole tuple here.
                    grads[i] = (filter_gradients(grads[i][0], filter_minimum), grads[i][1])
                if self._l2_constant and not self._l1_constant:
                    self._model.layers[i]._weights += self._learning_rate*grads[i][0] - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._weights
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]
                
                if self._l2_constant and self._l1_constant:
                    self._model.layers[i]._weights += self._learning_rate*grads[i][0] - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._weights - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._weights)
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]
                
                if self._l1_constant and not self._l1_constant:
                    self._model.layers[i]._weights += self._learning_rate*grads[i][0] - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._weights)
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]
                
                if not self._l1_constant and not self._l2_constant:
                    self._model.layers[i]._weights += self._learning_rate*grads[i][0]
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]
            
            if self._model._layer_type == 'Sparse':
                if self._l2_constant and not self._l1_constant:
                    self._model.layers[i]._weights.data += self._learning_rate*grads[i][0] - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._weights.data
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]
                
                if self._l2_constant and self._l1_constant:
                    self._model.layers[i]._weights.data += self._learning_rate*grads[i][0] - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._weights.data - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._weights.data)
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]
                
                if self._l1_constant and not self._l1_constant:
                    self._model.layers[i]._weights.data += self._learning_rate*grads[i][0] - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._weights.data)
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]
                
                if not self._l1_constant and not self._l2_constant:
                    self._model.layers[i]._weights.data += self._learning_rate*grads[i][0]
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]
            
class madadad_optimizer:
    
    """ Adadad is a kind of adaptive gradients optimizer, where gradients that keep moving in the same direction, move faster. """
    
    def __init__(self, model, learning_rate, friction, l1_constant = None, l2_constant = None):
        
        self._model = model
        self._learning_rate = learning_rate
        if self._model._comp_type == 'CPU':
            if self._model._layer_type == 'Full':
                self._adadad_weights = [np.zeros(i._weights.shape) for i in self._model.layers]
                self._adadad_biases = [np.zeros(i._biases.shape) for i in self._model.layers]
            if self._model._layer_type == 'Sparse':
                stats = []
        if self._model._comp_type == 'GPU':
            if self._model._layer_type == 'Full':
                self._adadad_weights = [cp.zeros(i._weights.shape) for i in self._model.layers]
                self._adadad_biases = [cp.zeros(i._biases.shape) for i in self._model.layers]
            if self._model._layer_type == 'Sparse':
                stats = []
        self._gradients = []
        self._friction = friction
        self._l2_constant = l2_constant
        self._l1_constant = l1_constant
    
    def train_step(self, inputs, labels):
        
        grads = get_gradients(self, inputs, labels)

        for i in range(len(grads)):
            signs = np.sign(grads[i][0])
            bias_signs = np.sign(grads[i][1])
            self._adadad_weights[i] = (np.sign(self._adadad_weights[i]) == signs)*self._adadad_weights[i]
            self._adadad_biases[i] = (np.sign(self._adadad_biases[i]) == bias_signs)*self._adadad_biases[i]
            self._adadad_weights[i] = self._friction*self._adadad_weights[i] + signs
            self._adadad_biases[i] = self._friction*self._adadad_biases[i] + bias_signs
            
        
        for i in range(len(grads)):
            if self._model._layer_type == 'Full':
                if self._l2_constant and not self._l1_constant:
                    self._model.layers[i]._weights += self._learning_rate*grads[i][0]*abs(self._adadad_weights[i]) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._weights
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]*abs(self._adadad_biases[i]) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._biases

                if self._l2_constant and self._l1_constant:
                    self._model.layers[i]._weights += self._learning_rate*grads[i][0]*abs(self._adadad_weights[i]) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._weights - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._weights)
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]*abs(self._adadad_biases[i]) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._biases - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._biases)
                
                if self._l1_constant and not self._l2_constant:
                    self._model.layers[i]._weights += self._learning_rate*grads[i][0]*abs(self._adadad_stats[i]) - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._weights)
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]*abs(self._adadad_biases[i]) -self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._biases)
                
                if not self._l1_constant and not self._l2_constant:
                    self._model.layers[i]._weights += self._learning_rate*grads[i][0]*abs(self._adadad_stats[i])
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]
                
            
            if self._model._layer_type == 'Sparse':
                self._model.layers[i]._weights = (self._model.layers[i]._weights + self._learning_rate*grads[i][0]).tocoo()
                self._model.layers[i]._biases += self._learning_rate*grads[i][1]

class adadad_optimizer:
    
    """ Adfm is a kind of adaptive gradients optimizer, where gradients that keep moving in the same direction, move faster. Modified from the adadad optimizer I first developed to include a friction constant and momentum parameter. 
    """
    
    def __init__(self, model, learning_rate, friction, momentum = None, l1_constant = None, l2_constant = None, epsilon = 1e-7):
        
        self._model = model
        self._learning_rate = learning_rate
        if self._model._comp_type == 'CPU':
            if self._model._layer_type == 'Full':
                self._adadad_weights = [np.zeros(i._weights.shape) for i in self._model.layers]
                self._adadad_biases = [np.zeros(i._biases.shape) for i in self._model.layers]
            if self._model._layer_type == 'Sparse':
                stats = []
        if self._model._comp_type == 'GPU':
            if self._model._layer_type == 'Full':
                self._adadad_weights = [cp.zeros(i._weights.shape) for i in self._model.layers]
                self._adadad_biases = [cp.zeros(i._biases.shape) for i in self._model.layers]
            if self._model._layer_type == 'Sparse':
                stats = []
        self._gradients = []
        self._l2_constant = l2_constant
        self._l1_constant = l1_constant
        self._friction = friction
        self._momentum = momentum
        self._epsilon = epsilon
        self._steps = 0
            
    def train_step(self, inputs, labels):
        self._steps += 1
        grads = get_gradients(self, inputs, labels)

        for i in range(len(grads)):
            if self._model._layer_type == 'Full':
                if self._l2_constant and not self._l1_constant:
                    self._model.layers[i]._weights += self._learning_rate*(grads[i][0] + self._adadad_weights[i]) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._weights
                    self._model.layers[i]._biases += self._learning_rate*(grads[i][1] + self._adadad_biases[i]) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._biases
                
                if self._l2_constant and self._l1_constant:
                    self._model.layers[i]._weights += self._learning_rate*(grads[i][0] + self._adadad_weights[i]) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._weights - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._weights)
                    self._model.layers[i]._biases += self._learning_rate*(grads[i][1] + self._adadad_biases[i]) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._biases - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._biases)
                
                if self._l1_constant and not self._l2_constant:
                    self._model.layers[i]._weights += self._learning_rate*(grads[i][0] + self._adadad_stats[i]) - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._weights)
                    self._model.layers[i]._biases += self._learning_rate*(grads[i][1] + self._adadad_biases[i]) - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._biases)
                
                if not self._l1_constant and not self._l2_constant:
                    self._model.layers[i]._weights += self._learning_rate*(grads[i][0] + self._adadad_stats[i])
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]               
            
            if self._model._layer_type == 'Sparse':
                self._model.layers[i]._weights = (self._model.layers[i]._weights + self._learning_rate*grads[i][0]).tocoo()
                self._model.layers[i]._biases += self._learning_rate*grads[i][1]
            
        for i in range(len(grads)):
            #signs = np.sign(grads[i][0])
            squares = grads[i][0]*self._adadad_weights[i]
            bias_squares = grads[i][1]*self._adadad_biases[i]
            #self._adadad_stats[i] = (np.sign(self._adadad_stats[i][0]) == signs)*self._adadad_stats[i][0]
            self._adadad_weights[i] = (squares > -self._epsilon)*self._adadad_weights[i]
            self._adadad_biases[i] = (bias_squares > -self._epsilon)*self._adadad_biases[i]
            self._adadad_weights[i] = self._adadad_weights[i]*self._friction + grads[i][0]
            self._adadad_biases[i] = self._adadad_biases[i]*self._friction + grads[i][1]
            if self._steps < 10:
                self._adadad_weights[i] *= 0 
                self._adadad_biases[i] *= 0 
    
class adam_optimizer:
    
    """ Adam optimizer with quadratic cost function"""
    def __init__(self, model, learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8, l1_constant = None, l2_constant = None, bitmasks = None, backward_layers = 0):
        self._model = model
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._gradients = []
        self._l2_constant = l2_constant
        self._l1_constant = l1_constant
        self._backward_layers = backward_layers
        if self._model._layer_type == 'Sparse':
            self._sparse_coords = []
            for layer in self._model.layers:
                temp = layer._weights.tocoo()
                self._sparse_coords.append(cp.array([(i, j) for i, j in zip(cp.asnumpy(temp.row), cp.asnumpy(temp.col))]))
                temp = None
        if self._model._comp_type == 'CPU':
            if self._model._layer_type == 'Full':
                self._weight_m1 = [np.zeros(i._weights.shape) for i in self._model.layers[-self._backward_layers:]]
                self._bias_m1 = [np.zeros(i._biases.shape) for i in self._model.layers[-self._backward_layers:]]
                self._weight_m2 = [np.zeros(i._weights.shape) for i in self._model.layers[-self._backward_layers:]]
                self._bias_m2 = [np.zeros(i._biases.shape) for i in self._model.layers[-self._backward_layers:]]
            if self._model._layer_type == 'Sparse':
                print('Sparse Adam not implemented yet')
                stats = []
        if self._model._comp_type == 'GPU':
            if self._model._layer_type == 'Full':
                self._weight_m1 = [cp.zeros(i._weights.shape) for i in self._model.layers[-self._backward_layers:]]
                self._bias_m1 = [cp.zeros(i._biases.shape) for i in self._model.layers[-self._backward_layers:]]
                self._weight_m2 = [cp.zeros(i._weights.shape) for i in self._model.layers[-self._backward_layers:]]
                self._bias_m2 = [cp.zeros(i._biases.shape) for i in self._model.layers[-self._backward_layers:]]
            if self._model._layer_type == 'Sparse':
                self._weight_m1 = [cp.zeros((i._weights.nnz)) for i in self._model.layers]
                self._bias_m1 = [cp.zeros(i._biases.shape) for i in self._model.layers]
                self._weight_m2 = [cp.zeros((i._weights.nnz)) for i in self._model.layers]
                self._bias_m2 = [cp.zeros(i._biases.shape) for i in self._model.layers]
        self._timestep = 0
        # Bitmasks for training only on selected parameters, input as full matrices stored on the same device as the weight matrix.
        self._bitmasks = bitmasks
        
    def train_step(self, inputs, labels, train_biases = True, filter_minimum = None):
        if self._model._layer_type == 'Full':
            
            # Backward layers here in get gradients
            # I think the weight and bias moment updates whould work the same
            # The loop updating weights needs to be modified... 
            
            if self._backward_layers == 0:
                grads = get_gradients(self, inputs, labels)
            else:
                grads = get_gradients(self, inputs, labels, backward_layers = self._backward_layers)
                if filter_minimum:
                    # I need to replace the whole tuple here.
                    grads[i] = (filter_gradients(grads[i][0], filter_minimum), grads[i][1])
        if self._model._layer_type == 'Sparse':
            grads = get_gradients(self, inputs, labels, self._sparse_coords)
        
        grads = [(np.clip(i[0], -1, 1), np.clip(i[1], -1, 1)) for i in grads]
        if self._bitmasks:
            for i in range(len(grads)):
                grads[i][0] = grads[i][0]*self._bitmask[i]
        self._timestep += 1
        co_learning_rate = self._learning_rate*(np.sqrt(1 - self._beta2**self._timestep)/(1-self._beta1**self._timestep))
                
        for i in range(len(grads)):
            self._weight_m1[i] = self._beta1*self._weight_m1[i] + (1-self._beta1)*grads[i][0]
            self._bias_m1[i] = self._beta1*self._bias_m1[i] + (1-self._beta1)*grads[i][1]                                  
            self._weight_m2[i] = self._beta2*self._weight_m2[i] + (1-self._beta2)*grads[i][0]*grads[i][0]
            self._bias_m2[i] = self._beta2*self._bias_m2[i] + (1-self._beta2)*grads[i][1]*grads[i][1]
                                                
                        
            if self._model._layer_type == 'Full':
                if self._l2_constant and not self._l1_constant:
                    self._model.layers[self._model._depth - len(grads) + i]._weights += co_learning_rate*self._weight_m1[i]/(np.sqrt(self._weight_m2[i])+self._epsilon) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._weights
                    if train_biases:
                        self._model.layers[self._model._depth - len(grads) + i]._biases += co_learning_rate*self._bias_m1[i]/(np.sqrt(self._bias_m2[i])+self._epsilon) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._biases
                
                if self._l2_constant and self._l1_constant:
                    self._model.layers[self._model._depth - len(grads) + i]._weights += co_learning_rate*self._weight_m1[i]/(np.sqrt(self._weight_m2[i])+self._epsilon) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._weights - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._weights)
                    if train_biases:
                        self._model.layers[self._model._depth - len(grads) + i]._biases += co_learning_rate*self._bias_m1[i]/(np.sqrt(self._bias_m2[i])+self._epsilon) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._biases - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._biases)
                
                if self._l1_constant and not self._l1_constant:
                    self._model.layers[self._model._depth - len(grads) + i]._weights += co_learning_rate*self._weight_m1[i]/(np.sqrt(self._weight_m2[i])+self._epsilon) - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._weights)
                    if train_biases:
                        self._model.layers[self._model._depth - len(grads) + i]._biases += co_learning_rate*self._bias_m1[i]/(np.sqrt(self._bias_m2[i])+self._epsilon) -  self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._biases)
                
                if not self._l1_constant and not self._l2_constant:
                    self._model.layers[self._model._depth - len(grads) + i]._weights += co_learning_rate*self._weight_m1[i]/(np.sqrt(self._weight_m2[i])+self._epsilon)
                    if train_biases:
                        self._model.layers[self._model._depth - len(grads) + i]._biases += co_learning_rate*self._bias_m1[i]/(np.sqrt(self._bias_m2[i])+self._epsilon)
            
            if self._model._layer_type == 'Sparse':
                if self._l2_constant and not self._l1_constant:
                    self._model.layers[i]._weights.data += self._learning_rate*grads[i][0] - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._weights.data
                
                if self._l2_constant and self._l1_constant:
                    self._model.layers[i]._weights.data += self._learning_rate*grads[i][0] - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[i]._weights.data - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._weights.data)
                
                if self._l1_constant and not self._l1_constant:
                    self._model.layers[i]._weights.data += self._learning_rate*grads[i][0] - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[i]._weights.data)
                
                if not self._l1_constant and not self._l2_constant:
                    self._model.layers[i]._weights.data += self._learning_rate*grads[i][0]
                    self._model.layers[i]._biases += self._learning_rate*grads[i][1]
                    
class selected_adam_optimizer:
    
    """ Adam optimizer with quadratic cost function. This one optimizes over a selection of parameters, not optimized for speed yet, just using bitmasks and such. Inputs a list of parameters that will be updated."""
    def __init__(self, model, learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8, l1_constant = None, l2_constant = None, backward_layers = None, train_final_layer = False):
        self._model = model
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._gradients = []
        self._l2_constant = l2_constant
        self._l1_constant = l1_constant
        if backward_layers == None:
            self._backward_layers = len(self._model.layers)
        else:
            self._backward_layers = backward_layers
        self._train_final_layer = train_final_layer
        if train_final_layer:
            self._backward_layers = 1
        self._weight_m1 = []
        self._bias_m1 = []
        self._weight_m2 = []
        self._bias_m2 = []
        if self._model._comp_type == 'CPU':
            if self._model._layer_type == 'Full':
                for i in range(self._backward_layers):
                    self._weight_m1.append(np.zeros(self._model.layers[i+len(self._model.layers) - self._backward_layers]._weights.shape))
                    self._bias_m1.append(np.zeros(self._model.layers[i+ len(self._model.layers) - self._backward_layers]._biases.shape))
                    self._weight_m2.append(np.zeros(self._model.layers[i+ len(self._model.layers) - self._backward_layers]._weights.shape))
                    self._bias_m2.append(np.zeros(self._model.layers[i+ len(self._model.layers) - self._backward_layers]._biases.shape))
            if self._model._layer_type == 'Sparse':
                print('Sparse Adam not implemented yet')
                stats = []
        if self._model._comp_type == 'GPU':
            if self._model._layer_type == 'Full':
                for i in range(self._backward_layers):
                    self._weight_m1.append(cp.zeros(self._model.layers[i+len(self._model.layers) - self._backward_layers]._weights.shape))
                    self._bias_m1.append(cp.zeros(self._model.layers[i + len(self._model.layers) - self._backward_layers]._biases.shape))
                    self._weight_m2.append(cp.zeros(self._model.layers[i+len(self._model.layers) - self._backward_layers]._weights.shape))
                    self._bias_m2.append(cp.zeros(self._model.layers[i+len(self._model.layers) - self._backward_layers]._biases.shape))
            if self._model._layer_type == 'Sparse':
                stats = []
                print('Sparse Adam not implemented yet')
        self._timestep = 0
        self._layers = None
                
    def train_step(self, inputs, labels, layers = None, train_biases = True):
        
        grads = get_gradients(self, inputs, labels, backward_layers = self._backward_layers)
                        
        grads = [(np.clip(i[0], -1, 1), np.clip(i[1], -1, 1)) for i in grads]
        # Multiply gradients by the sparse bitmasks
        if self._train_final_layer == False:
            for i in range(len(grads)):
                
                grads[-(i+1)] = (np.multiply(grads[-(i+1)][0], self._model.layers[-(i+1)]._sparse_training_mask), grads[-(i+1)][1])
        
        self._timestep += 1
        
        co_learning_rate = self._learning_rate*(np.sqrt(1 - self._beta2**self._timestep)/(1-self._beta1**self._timestep))
                
        for i in range(self._backward_layers):
            self._weight_m1[-(i+1)] = self._beta1*self._weight_m1[-(i+1)] + (1-self._beta1)*grads[-(i+1)][0]                            
            self._weight_m2[-(i+1)] = self._beta2*self._weight_m2[-(i+1)] + (1-self._beta2)*grads[-(i+1)][0]*grads[-(i+1)][0]
            if train_biases:
                self._bias_m1[-(i+1)] = self._beta1*self._bias_m1[-(i+1)] + (1-self._beta1)*grads[-(i+1)][1]    
                self._bias_m2[-(i+1)] = self._beta2*self._bias_m2[-(i+1)] + (1-self._beta2)*grads[-(i+1)][1]*grads[-(i+1)][1]

            
            # Use loop to go forwards with the grads/moments arrays and backwards with the layers list. 
            
            if self._model._layer_type == 'Full':
                if self._l2_constant and not self._l1_constant:
                    self._model.layers[-(i+1)]._weights += co_learning_rate*self._weight_m1[-(i+1)]/(np.sqrt(self._weight_m2[i])+self._epsilon) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[-(i+1)]._weights
                    if train_biases:
                        self._model.layers[-(i+1)]._biases += co_learning_rate*self._bias_m1[-(i+1)]/(np.sqrt(self._bias_m2[-(i+1)])+self._epsilon) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[-(i+1)]._biases
                
                if self._l2_constant and self._l1_constant:
                    self._model.layers[-(i+1)]._weights += co_learning_rate*self._weight_m1[-(i+1)]/(np.sqrt(self._weight_m2[-(i+1)])+self._epsilon) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[-(i+1)]._weights - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[-(i+1)]._weights)
                    if train_biases:
                        self._model.layers[-(i+1)]._biases += co_learning_rate*self._bias_m1[-(i+1)]/(np.sqrt(self._bias_m2[-(i+1)])+self._epsilon) - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[-(i+1)]._biases - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[-(i+1)]._biases)
                
                if self._l1_constant and not self._l1_constant:
                    self._model.layers[-(i+1)]._weights += co_learning_rate*self._weight_m1[-(i+1)]/(np.sqrt(self._weight_m2[-(i+1)])+self._epsilon) - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[-(i+1)]._weights)
                    if train_biases:
                            self._model.layers[-(i+1)]._biases += co_learning_rate*self._bias_m1[-(i+1)]/(np.sqrt(self._bias_m2[-(i+1)])+self._epsilon) -  self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[-(i+1)]._biases)
                
                if not self._l1_constant and not self._l2_constant:
                    self._model.layers[-(i+1)]._weights += co_learning_rate*self._weight_m1[-(i+1)]/(np.sqrt(self._weight_m2[-(i+1)])+self._epsilon)
                    if train_biases:
                        self._model.layers[-(i+1)]._biases += co_learning_rate*self._bias_m1[-(i+1)]/(np.sqrt(self._bias_m2[-(i+1)])+self._epsilon)
            
            if self._model._layer_type == 'Sparse':
                if self._l2_constant and not self._l1_constant:
                    self._model.layers[-(i+1)]._weights.data += self._learning_rate*grads[-(i+1)][0] - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[-(i+1)]._weights.data
                
                if self._l2_constant and self._l1_constant:
                    self._model.layers[-(i+1)]._weights.data += self._learning_rate*grads[-(i+1)][0] - self._l2_constant/inputs.shape[0]*self._learning_rate*self._model.layers[-(i+1)]._weights.data - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[-(i+1)]._weights.data)
                
                if self._l1_constant and not self._l1_constant:
                    self._model.layers[-(i+1)]._weights.data += self._learning_rate*grads[-(i+1)][0] - self._l1_constant/inputs.shape[0]*self._learning_rate*np.sign(self._model.layers[-(i+1)]._weights.data)
                
                if not self._l1_constant and not self._l2_constant:
                    self._model.layers[-(i+1)]._weights.data += self._learning_rate*grads[-(i+1)][0]
                    self._model.layers[-(i+1)]._biases += self._learning_rate*grads[-(i+1)][1]
        return grads
        
class subnet_finder:
    
    """ First attempt at building an optimizer, only uses quadratic cost function. There are parts to this that could be built into 
    othere sections of the library, but I don't know if this will work, and they might just end up bloating the files."""
    def __init__(self, model, error_mean = None):
        self._model = model
        self._alpha = None
        self._subnet_size = None
        # Initialize subnet masks, reset with each step
        self._subnet_masks = [np.zeros(i._weights.shape) for i in self._model.layers]
        # Scores initialized at 0, starting point is arbitrary since they are all sorted at the end
        self._weight_scores = [cp.zeros(i._weights.shape) for i in self._model.layers]
        self._error_mean = error_mean
        self._steps = 0
        self._above_threshold = 0


    def random_train_step(self, inputs, labels, ratio = None, error_type = 'quadratic'):
        
        ''' Forward pass with the dropout mask chosen randomly'''
        
        if self._model._layer_type == 'Full':
            this_layer_inputs = inputs
        if self._model._comp_type == 'GPU':
            this_layer_inputs = cp.array(this_layer_inputs)
        
        for layer in self._model._layers:
            outputs = layer.activate_NG(inputs = this_layer_inputs, ratio = ratio,  distribution = 'binomial')
            this_layer_inputs = outputs
        
        if self._model._comp_type == 'CPU':
            if error_type == 'quadratic':
                error = np.mean((outputs - labels)**2)
            if error_type == 'argmax':
                error = np.mean(np.argmax(outputs, axis = 1) == np.argmax(labels, axis = 1))
        if self._model._comp_type == 'GPU':
            if error_type == 'quadratic':
                error = np.mean((outputs - cp.array(labels))**2)
            if error_type == 'argmax':
                error = np.mean(np.argmax(outputs, axis = 1) == np.argmax(cp.array(labels), axis = 1))

        # Update error mean
        if error_type == 'quadratic':
            if self._steps == 0 :
                self._error_mean = error
            else:
                delta = error - self._error_mean
                self._error_mean += delta/self._steps
                for i in range(len(self._weight_scores)):
                    self._weight_scores[i] += self._model._layers[i]._dropout_mask*delta 
            self._steps += 1
        if error_type == 'argmax':
            # Use the error mean as the expected value from a random selection
            #self._error_mean = 1/self._model.layers[-1]._size
            if error > self._error_mean:
                self._above_threshold += 1
                for i in range(len(self._weight_scores)):
                        self._weight_scores[i] += self._model._layers[i]._dropout_mask*(error - self._error_mean) 
    
    def gaussian_train_step(self, inputs, labels, temp):
        ''' More targeted parameter selection'''
        
        print('Do the thing')
               
    def get_accuracy(self, inputs, labels):
        if self._model._layer_type == 'Full':
            this_layer_inputs = inputs
        if self._model._comp_type == 'GPU':
            this_layer_inputs = cp.array(this_layer_inputs)
        
        for layer in self._model._layers:
            outputs = layer.activate_NG(inputs = this_layer_inputs, ratio = None,  distribution = None)
            this_layer_inputs = outputs
            
        return np.mean(np.argmax(outputs, axis = 1) == np.argmax(cp.array(labels), axis = 1))
    
    def choose_parameters(self, parameter_ratio, layers = None):
        ''' Sets the mask as ratio% of parameters with the highest scores.'''
        if layers:
            for i in layers:
                # get indices
                indices = get_k_max([self._weight_scores[i]], parameter_ratio)
                # make the mask
                if self._model._comp_type == 'GPU':
                    self._model.layers[i]._dropout_mask = cp.zeros(self._model.layers[i]._weights.shape)
                if self._model._comp_type == 'CPU':
                    self._model.layers[i]._dropout_mask = np.zeros(self._model.layers[i]._weights.shape)
                for j in indices:
                    
                    self._model.layers[i]._dropout_mask[j[0]][j[1]] = 1
                    
            print('You have set the bitmasks for ', layers,' do not forget to set the rest')
        else:
            for i in range(len(self._weight_scores)):
                # Get indices
                indices = get_k_max([self._weight_scores[i]], parameter_ratio)
                # Make the mask
                if self._model._comp_type == 'GPU':
                    self._model.layers[i]._dropout_mask = cp.zeros(self._model.layers[i]._weights.shape)
                if self._model._comp_type == 'CPU':
                    self._model.layers[i]._dropout_mask = np.zeros(self._model.layers[i]._weights.shape)
                for j in indices[0]:
                    self._model.layers[i]._dropout_mask[j[0]][j[1]] = 1
        return 
    
    def set_ones_bitmask(self, layers):
        '''Sets the bitmasks in the given layers to ones'''
        for i in layers:
            if self._model._comp_type == 'GPU':
                self._model.layers[i]._dropout_mask = cp.ones(self._model.layers[i]._weights.shape)
            if self._model._comp_type == 'CPU':
                self._model.layers[i]._dropout_mask = np.ones(self._model.layers[i]._weights.shape)