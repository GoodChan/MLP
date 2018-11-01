from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import random
import copy
import numpy as np
import sys
import matplotlib.pyplot as plt

class NeuralNetLearner(SupervisedLearner):
    
    def __init__(self):
        self.weights = np.array([])
        self.delta_weights = []
        self.mse = []
        
        # hand-picked constants
        self.num_nodes = [8, 10, 10]
        self.training_set_percentage = .80
        self.learning_rate = .05
        self.use_momentum = True
        self.count_down = 7
        #momentum
        self.alpha_num = .9
        
    def forward_pass(self, data_in, net):
        data = np.append(data_in, 1)
        net.append(data)
        for l in range(len(self.weights)): 
            temp_net = np.matmul(self.weights[l], data)
            #apply sigmoid
            temp_net = 1 / (1 + np.e ** -temp_net)
            if l < len(self.weights) - 1:
                temp_net = np.append(temp_net, 1)
            data = temp_net
            net.append(temp_net)
        return temp_net
        
    def backward_pass(self, features, label, output, net, alpha):
        error = []
        new_weights = []
        back = len(self.num_nodes) - 1
        
        for l in range(len(self.num_nodes)):
            fprime_net = net[back + 1] * (1 - net[back + 1])
            if l == 0:
                # ouput node
                #delta = (tj -Oj) f'(net)
                error.append((label - output) * fprime_net)
            else:
                #delta = sum of k(errork * weightjk) f'(net)
                error.append((np.matmul(self.weights[-l].T, error[l - 1]) * fprime_net)[:self.num_nodes[back]])
            #new_weights = learning_rate * Oi * errorj
            #C * outter product of net[back] and error[l]
            new_weights.append(self.learning_rate * (np.outer(error[l], net[back])) + (alpha * self.delta_weights[l]))
            back = back - 1
        self.delta_weights = new_weights
        #after all layers are calculated update weights
        for i, j in enumerate(reversed(range(len(self.weights)))):
            self.weights[i] = self.weights[i] + new_weights[j]

    def init_array(self, features):
        #+1 adds bias weights
        input_to_layer = len(features.data[0]) + 1
    
        layers = len(self.num_nodes)
        self.weights = [0] * layers 
        for i in range(layers):
            self.delta_weights.append(np.zeros((self.num_nodes[i], input_to_layer)))
            self.weights[i] = np.random.random((self.num_nodes[i], input_to_layer))-0.5
            input_to_layer = self.num_nodes[i] + 1
        self.delta_weights.reverse()
    
    def train(self, features, labels):
        alpha = 0
        if self.use_momentum:
            alpha = self.alpha_num;
        self.num_nodes.append(len(labels.enum_to_str[0]))
        self.init_array(features)
        previous_mse = 0

        features.shuffle(labels)
        num_train = int(len(features.data) * self.training_set_percentage)
        training_data = Matrix(features, 0, 0, num_train, features.cols)
        training_labels = Matrix(labels, 0, 0, num_train, labels.cols)
        validation_data = Matrix(features, num_train, 0, features.rows - num_train, features.cols)
        validation_labels = Matrix(labels, num_train, 0, labels.rows - num_train, labels.cols)

        count = self.count_down
        bssf = []
        best_mse = 0.0
        training_mse = []
        training_mse.append(0.0)
        
        for l in range(1000):
            #stoping criteria
            curr_mse = self.stopping_criteria(validation_data, validation_labels)
            print("curr_mse VS: ", curr_mse)
            if curr_mse < best_mse:
                bssf = [x.copy() for x in self.weights]
                best_mse = curr_mse
            if (previous_mse - curr_mse) < 0.001:
                count -= 1
                if (count <= 0):
                    self.weights = bssf or self.weights
                    print("num epochs: ", l)
                    break
            else:
                count = self.count_down
            previous_mse = curr_mse
            
            features.shuffle(labels)
            for i in range(len(training_data.data)):
                net = []
                new_weights = []
                #vector of input features
                data_in = np.array(training_data.data[i])
                output = self.forward_pass(data_in, net)
                temp = np.zeros(output.shape)
                temp[int(training_labels.data[i][0])] = 1.0
                training_mse[-1] += np.sum((output - temp)**2)
                
                # one hot encoding
                temp = np.zeros(output.shape)
                temp[int(training_labels.data[i][0])] = 1.0
                self.backward_pass(features.data[i], temp, output, net, alpha)
            training_mse[-1] /= len(features.data)
            print("training_mse: ", training_mse[0])
            
    def stopping_criteria(self, features, labels):
        correct = 0
        self.mse.append(0.0)
        for i in range(len(features.data)):
            label = []
            output = self.predict(features.data[i], label)
            temp = np.zeros(output.shape)
            temp[int(labels.data[i][0])] = 1.0
            self.mse[-1] += np.sum((output - temp)**2)
            if label == labels.data[i]:
                correct += 1
        self.mse[-1] /= len(features.data)
        accuracy = correct / len(features.data)
        print("accuracy: ", accuracy)
        return self.mse[-1]
    
    def predict(self, features, labels):
        del labels[:]
        net = []
        output = self.forward_pass(features, net)
        
        # one hot encoding
        index_of_output = 0
        for i in range(len(output)):
            if output[i] > output[index_of_output]:
                index_of_output = i
        labels.append(index_of_output)
        return output
                                  
