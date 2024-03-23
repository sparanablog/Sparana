import numpy as np
import gzip

class loader:
    
    def __init__(self, training_data, training_labels, test_data, test_labels):
        ''' I will need to feed in extracted data, data might be stored in different formats.
        This is for shuffling and tracking training.'''
        self._training_data = training_data
        self._training_labels = training_labels
        self._test_data = test_data
        self._test_labels = test_labels
        self._index_list = np.arange(len(self._training_data))
        np.random.shuffle(self._index_list)
        # Now for some things to keep track of
        self._minibatches = 0
        self._epochs = 0
        self._minibatch_index = 0
        
    def random_minibatch(self, batch_size):
        ''' This just selects a random minibatch from the whole training set, doesn't track epochs'''
        np.random.shuffle(self._index_list)
        data = self._training_data[self._index_list[:batch_size]]
        labels = self._training_labels[self._index_list[:batch_size]]
        return data, labels
    
    def minibatch(self, batch_size):
        ''' This takes minibatches from a shuffled training data set, tracks epochs, 
        and reshuffles the training data when the epoch is complete'''
        if self._minibatch_index + batch_size > len(self._training_data):
            np.random.shuffle(self._index_list)
            self._epochs += 1
            self._minibatch_index = 0
        data = self._training_data[self._index_list[self._minibatch_index : self._minibatch_index + batch_size]]
        labels = self._training_labels[self._index_list[self._minibatch_index : self._minibatch_index + batch_size]]
        self._minibatch_index += batch_size
        self._minibatches += 1
        return data, labels
    
    def print_stats(self):
        ''' Prints any useful information'''
        print('Epochs: ', self._epochs)
        print('Minibatches: ', self._minibatches)
        return
    
    def test_data(self):
        '''This seems pretty pointless until I am loading files directly with this class.'''
        return self._test_data


    def test_labels(self):
        return self._test_labels
    
class split_loader:
    
    '''This splits the data for my experiments with different classes, to test things like transfer learning and catastrophic forgetting. Remove classes inputs a list of classes to be removed and saved for later, add_classes gives a list of how many columns to add to the labels. For example. I want to remove 2 classes from the main dataset, and add one class back to try and train on both the removed classes with a different subset of parameters. Maintain_classes keeps all of the labels in the same place, just removes them.'''
    
    def __init__(self, training_data, training_labels, test_data, test_labels, remove_classes, add_classes = None, maintain_classes = False):
        ''' I will need to feed in extracted data, data might be stored in different formats.
        This is for shuffling and tracking training.'''
        self._training_data = training_data
        self._training_labels = training_labels
        self._test_data = test_data
        self._test_labels = test_labels
        if type(remove_classes) == int:
            self._remove_classes = [remove_classes]
        else:
            self._remove_classes = remove_classes
        self._add_classes = add_classes
        self._removed_training_indices = []
        self._removed_test_indices = []
        self._maintain_classes = maintain_classes
        # Removed classes are stored individually in a list e.g. [[2s],[3s]]
        self._removed_training_data = []
        self._removed_test_data = []
        # Find the indices of the classes, and put the datapoints in the removed classes list
        for i in enumerate(remove_classes):
            self._removed_training_indices.append([j[0] for j in enumerate(self._training_labels) if np.argmax(j[1])==i[1]])
            self._removed_training_data.append(self._training_data[self._removed_training_indices[i[0]]])
            self._removed_test_indices.append([j[0] for j in enumerate(self._test_labels) if np.argmax(j[1])==i[1]])
            self._removed_test_data.append(self._test_data[self._removed_test_indices[i[0]]])
        # Delete removed classes based on the indices found above
        self._training_data = np.delete(self._training_data, np.concatenate(self._removed_training_indices), axis = 0)
        self._training_labels = np.delete(self._training_labels, np.concatenate(self._removed_training_indices), axis = 0)
        self._test_data = np.delete(self._test_data, np.concatenate(self._removed_test_indices), axis = 0)
        self._test_labels = np.delete(self._test_labels, np.concatenate(self._removed_test_indices), axis = 0)
        # Delete the columns of 0s from the labels array
        if maintain_classes == False:
            self._training_labels = np.delete(self._training_labels, self._remove_classes, axis = 1)
            self._test_labels = np.delete(self._test_labels, self._remove_classes, axis = 1)
        # Get the number of output classes to reconstruct labels
        self._output_classes = self._test_labels.shape[1]
        # Parameters for randomizing minibatches
        self._index_list = np.arange(len(self._training_data))
        self._removed_index_list = [np.arange(len(i)) for i in self._removed_training_data]
        np.random.shuffle(self._index_list)
        for i in self._removed_index_list:
            np.random.shuffle(i)
        # Now for some things to keep track of things
        self._minibatches = 0
        self._removed_minibatches = [0 for i in self._removed_index_list]
        self._epochs = 0
        self._removed_epochs = [0 for i in self._removed_index_list]
        self._minibatch_index = 0
        self._removed_index = 0
        self._removed_minibatch_index = [0 for i in self._removed_index_list]
        
    def random_minibatch(self, batch_size):
        ''' This just selects a random minibatch from the whole training set, doesn't track epochs'''
        np.random.shuffle(self._index_list)
        data = self._training_data[self._index_list[:batch_size]]
        labels = self._training_labels[self._index_list[:batch_size]]
        return data, labels
    
    def removed_random_minibatch(self, batch_size, index = 0, label_index=-1):
        
        np.random.shuffle(self._removed_index_list[index])
        data = self._removed_training_data[index][self._index_list[index][:batch_size]]
        
        labels = np.zeros((batch_size, self._output_classes))
        if self._maintain_classes:
            label_index = self._remove_classes
        labels[:, label_index] = 1
        return data, labels
    
    def minibatch(self, batch_size):
        ''' This takes minibatches from a shuffled training data set, tracks epochs, 
        and reshuffles the training data when the epoch is complete'''
        if self._minibatch_index + batch_size > len(self._training_data):
            np.random.shuffle(self._index_list)
            self._epochs += 1
            self._minibatch_index = 0
        data = self._training_data[self._index_list[self._minibatch_index : self._minibatch_index + batch_size]]
        labels = self._training_labels[self._index_list[self._minibatch_index : self._minibatch_index + batch_size]]
        # Add a column of zeros to the labels array
        if self._add_classes:
            labels = np.append(labels, np.zeros((batch_size, self._add_classes)), axis = 1)
        self._minibatch_index += batch_size
        self._minibatches += 1
        return data, labels
    
    def removed_minibatch(self, batch_size, index = 0, label_index = -1):
        ''' This takes minibatches from a removed training data set, tracks epochs, 
        and reshuffles the training data when the epoch is complete. Label is added to the last column by default.'''
        if self._maintain_classes:
            label_index = self._remove_classes[index]
        if type(index) == list:
            # Do the thing
            if type(label_index) != list:
                print('Set the label indices fool')
                return
            data = None
            labels = None
            for i in zip(index, label_index):
                this_data = self._removed_training_data[i[0]][self._removed_index_list[i[0]][self._removed_minibatch_index[i[0]] : self._removed_minibatch_index[i[0]] + int(batch_size/len(index))]]
                these_labels = np.zeros((int(batch_size/len(index)), self._output_classes))
                these_labels[:, i[1]] = 1
                if type(data) != np.ndarray:
                    data = this_data
                    labels = these_labels
                else:
                    data = np.concatenate((data, this_data))
                    labels = np.concatenate((labels, these_labels))
                self._removed_minibatch_index[i[0]] += int(batch_size/len(index))
            self._removed_minibatches[i[0]] += 1
        else:
            if self._removed_minibatch_index[index] + batch_size > len(self._removed_training_data[index]):
                np.random.shuffle(self._index_list)
                self._removed_epochs[index] += 1
                self._removed_minibatch_index[index] = 0
            data = self._removed_training_data[index][self._removed_index_list[index][self._removed_minibatch_index[index] : self._removed_minibatch_index[index] + batch_size]]
            labels = np.zeros((batch_size, self._output_classes))
            labels[:, label_index] = 1
            self._removed_minibatch_index[index] += batch_size
        
        return data, labels
    
    def mixed_minibatch(self, batch_size, index, removed_ratio, label_index = -1):
        '''Mixes a removed class with remaining classes, removed ratio is the ratio of the removed class, eg 0.8 is 80% removed class, gets shuffled again, just to mix up the removed class.'''
        data, labels = self.minibatch(int(batch_size*(1-removed_ratio)))
        removed_data, removed_labels = self.removed_minibatch(int(batch_size*removed_ratio), index, label_index)
        data = np.append(data, removed_data, axis = 0)
        labels = np.append(labels, removed_labels, axis = 0)
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        return data[indices], labels[indices]
        
    
    def print_stats(self):
        ''' Prints any useful information'''
        print('Epochs: ', self._epochs)
        print('Minibatches: ', self._minibatches)
        return
    
    def test_data(self):
        '''YAAY this is useful now'''
        return self._test_data

    def removed_test_set(self, index = 0, label_index = -1):
        if self._maintain_classes:
            label_index = self._remove_classes[index]
        if type(index) == list:
            # do the thing
            data = None
            labels = None
            for i in zip(index, label_index):
                this_data = self._removed_test_data[i[0]]
                these_labels = np.zeros((this_data.shape[0], self._output_classes))
                these_labels[:, i[1]] = 1
                if type(data) != np.ndarray:
                    data = this_data
                    labels = these_labels
                else:
                    data = np.concatenate((data, this_data))
                    labels = np.concatenate((labels, these_labels))
                
        else:
            data = self._removed_test_data[index]
            labels = np.zeros((data.shape[0], self._output_classes))
            labels[:, label_index] = 1
        return data, labels
    

    def test_labels(self):
        return self._test_labels
    
class distributed_loader:
    
    '''This splits a training set into a given number of subsets, these subsets can be called using an index. Test data is preserved.'''
    
    def __init__(self, training_data, training_labels, test_data, test_labels, subsets):
        ''' I will need to feed in extracted data, data might be stored in different formats.
        This is for shuffling and tracking training.'''
        self._test_data = test_data
        self._test_labels = test_labels
        # Get a randomised list
        self._index_list = np.arange(len(training_data))
        np.random.shuffle(self._index_list)
        # Shuffle and split the data
        self._training_data = np.array_split(training_data[self._index_list], subsets)
        self._training_labels = np.array_split(training_labels[self._index_list], subsets)
        # Get randomised lists for each subset
        self._index_list = [np.arange(len(i)) for i in self._training_data]
        for i in self._index_list:
            np.random.shuffle(i)
        # Now for some things to keep track of
        self._minibatches = [0 for i in self._training_data]
        self._epochs = [0 for i in self._training_data]
        self._minibatch_index = [0 for i in self._training_data]
        
    def random_minibatch(self, batch_size):
        ''' This just selects a random minibatch from the whole training set, doesn't track epochs'''
        np.random.shuffle(self._index_list)
        data = self._training_data[self._index_list[:batch_size]]
        labels = self._training_labels[self._index_list[:batch_size]]
        return data, labels
    
    def minibatch(self, batch_size, subset):
        ''' This takes minibatches from a shuffled training data set, tracks epochs, 
        and reshuffles the training data when the epoch is complete'''
        if self._minibatch_index[subset] + batch_size > len(self._training_data[subset]):
            np.random.shuffle(self._index_list[subset])
            self._epochs[subset] += 1
            self._minibatch_index[subset] = 0
            
        data = self._training_data[subset][self._index_list[subset][self._minibatch_index[subset] : self._minibatch_index[subset] + batch_size]]
        labels = self._training_labels[subset][self._index_list[subset][self._minibatch_index[subset] : self._minibatch_index[subset] + batch_size]]
        self._minibatch_index[subset] += batch_size
        self._minibatches[subset] += 1
        return data, labels
    
    def print_stats(self):
        ''' Prints any useful information'''
        print('Epochs: ', self._epochs)
        print('Minibatches: ', self._minibatches)
        return
    
    def combine_subsets(self, subsets):
        '''Combines the first X subsets for a bigger subset.'''
        # Combine the initia X(subsets) number of subsets
        othersubsets = self._training_data[(subsets+1):]
        
        for i in range(subsets):
            # Add the subsets here
            
            return
        # Then append othersubsets to self._training_data and shit
        
        #Then reset all of the tracking variables here
        
        
        return
    
    def test_data(self):
        '''This seems pretty pointless until I am loading files directly with this class.'''
        return self._test_data


    def test_labels(self):
        return self._test_labels
    