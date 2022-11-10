#from Neuron import Neuron
from Layer import Layer
import math
import json
import numpy as np
from copy import deepcopy

# tanh
def tanh(sum_of_products):
    return math.tanh(sum_of_products)

def rounded_sigmoid(sum_of_products, lambda_=0.8):
    ''' Created just to make results more similar to provided example where
    numbers were rounded on purpose for easier presentation on paper. '''
    return round(1 / (1 + math.exp(-lambda_*sum_of_products)), 2)

def sigmoid(sum_of_products, lambda_=0.8):
    return 1 / (1 + math.exp(-lambda_*sum_of_products))

def linear(sum_of_products):
    return sum_of_products

def show_network(layers):
    for i, layer in enumerate(layers):
        print(f'Layer {i+1} neurons:')
        for neuron in layer.neurons:
            print(f'    activation_value={neuron.activation_value}')
            print(f'    weights={neuron.weights}')
            print(f'    error={neuron.error}')
            print()
        print()
    print('\n')

def cost(results, observations):
    ''' The purpose of cost function is to measure how good the network is at predicting. 
    The lower the cost, the better the network is at predicting. We can use this value to
    decide when to stop training the network. '''
    c = 0
    for r, o in zip(results, observations):
        # abs is used to make sure that the cost is always positive
        # otherwise a positive error would cancel out a negative error
        # (e.g. if one output neuron error is -2 and the other is +2, 
        # the total error would be 0 instead of 4)
        c += abs(o - r)
        #c += (o - r) **2 
    return c 
    #return c /2

# root mean squared error
def cost2(results, observations):
    c = 0
    for r, o in zip(results, observations):
        c += (o - r) **2 
    return math.sqrt(c / len(results))

## binary cross entropy cost function
#def binary_cross_entropy_cost(results, observations):
#    # https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
#    c = 0
#    for r, o in zip(results, observations):
#        c += (o * math.log(r) + (1 - o) * math.log(1 - r))
#    return c
#
## quadratic cost function
#def quadratic_cost(results, observations):
#    c = 0
#    for r, o in zip(results, observations):
#        c += (o - r)**2
#    return c / 2

class NeuralNetHolder:
    '''  x[0] = first input value to the first layer (e.g. integer or float)
         x = all first layer inputs (a list), initially called "input_row" in the provided code template
         X = multiple examples of first layer inputs (list of lists)

         y[0] = first output value of the last layer (e.g. integer or float)
         y = all last layer outputs (a list)
         Y = multiple examples of last layer outputs (list of lists)
         '''
    def __init__(self, learning_rate=0.8, momentum=0.1):#, layer_sizes=[1, 2, 2], neuron_weights_values=[[], [[0.6, 0.7], [0.8, 0.8]], [[0.4, 0.5, 0.6], [0.5, 0.7, 0.9]]]):
        super().__init__()

        self.learning_rate = learning_rate
        self.momentum = momentum

#        self.layers = []
#        for i, layer_size in enumerate(layer_sizes):
#            if i == 0:
#                # input layer
#                layer = Layer(prev_neurons_count=0, neurons_count=layer_size, add_bias=True)
#            elif i == len(layer_sizes)-1:
#                # output layer
#                layer = Layer(prev_neurons_count=layer_sizes[i-1]+1, neurons_count=layer_size, add_bias=False, activation_function=linear, neurons_weights_values=neuron_weights_values[i])
#            else:
#                layer = Layer(prev_neurons_count=layer_sizes[i-1]+1, neurons_count=layer_size, add_bias=True, activation_function=sigmoid, neurons_weights_values=neuron_weights_values[i])
#            self.layers.append(layer)

        # neurons_count is the number of neurons excluding the bias neuron
        # prev_neurons_count is the number of neurons in the previous layer including the bias
        #self.layers = [ 
        #    # input layer
        #    Layer(prev_neurons_count=0, neurons_count=2, add_bias=True), 

        #    # hidden layer
        #    Layer(prev_neurons_count=3, neurons_count=2, add_bias=True, activation_function=sigmoid, neurons_weights_values=[
        #        [0.6, 0.7], # hidden layer, 1st node weights
        #        [0.8, 0.8]  # hidden layer, 2nd node weights
        #    ]),

        #    # output layer
        #    Layer(prev_neurons_count=3, neurons_count=2, add_bias=False, activation_function=linear, neurons_weights_values=[
        #        [0.4, 0.5, 0.5], # output layer, 1st node weights
        #        [0.5, 0.7, 0.9]  # output layer, 2nd node weights
        #    ])
        #    ]

        self.layers = [ 
           # input layer
           Layer(prev_neurons_count=0, neurons_count=2, add_bias=True), 

           # hidden layer
           Layer(prev_neurons_count=3, neurons_count=12, add_bias=True, activation_function=sigmoid),

           # output layer
           Layer(prev_neurons_count=13, neurons_count=2, add_bias=False, activation_function=linear)
           ]
        self.LAST_LAYER_INDEX = len(self.layers) - 1

    def forward_propagation(self, x):
        ''' Forward propagation is the process of calculating the output of each neuron in the network.
        It is the process of making a prediction. Each neuron value is calculated by summing the products
        of the weights and the values of the neurons in the previous layer, and then supplying this sum
        to an activation function (e.g. sigmoid, Relu). "Linear" activation function means the sum itself
        becomes the activation value. '''
        # set first layer activation values to inputs to neural network
        for i, neuron in enumerate(self.layers[0].neurons[1:]): # omitting 1st neuron because it's 1 by default (bias)
            neuron.activation_value = x[i]

        # for each layer except the first one
        for i, layer in enumerate(self.layers[1:]):
            # for each neuron in a layer, set activation value 
            # (based on previous weights and previous layer activation values)
            for neuron in layer.neurons:
                # if neuron is a bias, we can assume it already has an activation value (1)
                # so we don'n need to set it based on previous neurons. "neuron.is_bias" check is used instead 
                # of "layer.neurons[1:0]" indexing because the last layer does not have a bias neuron. 
                if neuron.is_bias:
                    continue
                #import pdb; pdb.set_trace()
                neuron.activate(self.layers[i])

        # activation values of the last layer are the results of the neural network
        results = [ neuron.activation_value for neuron in self.layers[-1].neurons ]
        return results

    def backward_propagation(self, y):
        ''' Backward propagation is the process of calculating the error of each neuron in the network,
        and updating weights accordingly. It is letting know each neuron how much it contributed to the inaccurate
        prediction, so when the next prediction is made, the neuron will more accurately predict the output. '''
        # set errors of the last layer before beginning backpropagation
        for i, (neuron, o) in enumerate(zip(self.layers[-1].neurons, y)):
            # r = result (prediction), o = observation
            self.layers[-1].neurons[i].error = o - neuron.activation_value

        # backpropagation (updating weights based on their prediction accuracy)
        # for each layer except input layer
        for i in range(len(self.layers)-1, 0, -1):
            for j, neuron in enumerate(self.layers[i].neurons):
                # last layer error is measured by the difference between prediction and observation
                # for other layers, error is measured using the sum of errors (multiplied by weights) of neurons in the next layer
                if i == self.LAST_LAYER_INDEX:
                    neuron.error = y[j] - neuron.activation_value 
                else:
                    neuron.error =  self.learning_rate * neuron.activation_value * (1-neuron.activation_value) * sum(n.weights[j] * n.error for n in self.layers[i+1].neurons)
                
                for k, weight in enumerate(neuron.weights):
                    weight_change = self.learning_rate * neuron.error * self.layers[i-1].neurons[k].activation_value + self.momentum * neuron.last_weight_changes[k]
                    # last weight changes are used to calculate momentum
                    neuron.last_weight_changes[k] = weight_change
                    # weights are not updated immediately because we need to use them to calculate errors of neurons in the previous layers
                    neuron.new_weights[k] += weight_change

        # Latch all neurons weights
        for i in range(len(self.layers)-1, 0, -1):
            for neuron in self.layers[i].neurons:
                neuron.update_weights(epsilon_regularisation=0.001) #neuron.weights = list(neuron.new_weights)


    def train(self, X, Y, epochs=1):

        # x_mins = np.min(X, axis=0)
        # x_maxs = np.max(X, axis=0)
        # y_mins = np.min(Y, axis=0)
        # y_maxs = np.max(Y, axis=0)

        x_means = np.mean(X, axis=0)
        x_stds = np.std(X, axis=0)
        y_means = np.mean(Y, axis=0)
        y_stds = np.std(Y, axis=0)
        X_orig = deepcopy(X)
        Y_orig = deepcopy(Y)
        # X = self.min_max_normalization(deepcopy(X), x_mins, x_maxs)
        # Y = self.min_max_normalization(deepcopy(Y), y_mins, y_maxs)
        X = self.normalize(X, x_means, x_stds)
        #Y = self.normalize(Y, y_means, y_stds)
        #self.normalization_parameters = {'x_mins': x_mins.tolist(), 'x_maxs': x_maxs.tolist(), 'y_mins': y_mins.tolist(), 'y_maxs': y_maxs.tolist()}
        self.normalization_parameters = {'x_means': x_means.tolist(), 'x_stds': x_stds.tolist(), 'y_means': y_means.tolist(), 'y_stds': y_stds.tolist()}
        costs = []
        # epoch is a single pass through the entire training dataset
        for epoch_index in range(epochs):
            epoch_cost = 0
            for i, (x, y, y_orig) in enumerate(zip(X, Y, Y_orig)):
                # results = self.forward_propagation(x)
                y_pred = self.forward_propagation(x)
                results = y_pred
                #results = self.unnormalize(y_pred, y_means, y_stds)
                errors = [ o - r for r,o in zip(results, y) ]
                self.backward_propagation(y)

                # print('results =', results)
                # print('y =', y)
                # print('errors =', errors)
                # print('self.predict(x) =', self.predict(X_orig[i]))
                epoch_cost += abs(cost(results, y_orig) / len(X))
                # print('cost =', cost(results, y_orig))
            if epoch_index > 1 and epoch_cost > costs[-1]:
                self.learning_rate *= 0.8
            print(f'{epoch_index+1}. cost =', epoch_cost)
            costs.append(epoch_cost)
            #print(f'{epoch_index}, ', end='')
        # import pdb; pdb.set_trace()
        return costs
    
    def predict(self, x, verbose=True):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        # [
        #  x_dist,  - high value means goal is on the right
        #  y_dist,  - high value means goal is below
        #  y_speed, - high value means spaceship is moving down
        #  x_speed  - high value means spaceship is moving right
        # ]
        if type(x) == str:
            x = [float(x_) for x_ in x.split(',')]
            print('x = ', x)
            # import pdb; pdb.set_trace() 
        #return [0.1, 0.2]
        x_normalized = self.normalize([x], np.array(self.normalization_parameters['x_means']), np.array(self.normalization_parameters['x_stds']))[0]
        # import pdb; pdb.set_trace()
        # x_normalized = self.min_max_normalization([x], np.array(self.normalization_parameters['x_mins']), np.array(self.normalization_parameters['x_maxs']))[0]
        y_normalized = self.forward_propagation(x_normalized)
        #y = self.min_max_unnormalization(y_normalized, self.normalization_parameters['y_mins'], self.normalization_parameters['y_maxs'])
        #y = self.unnormalize([y_normalized], self.normalization_parameters['y_means'], self.normalization_parameters['y_stds'])[0]
        # if verbose:
        #     print('x =', x, 'x_normalized =', x_normalized)
        #     print('y =', y, 'y_normalized =', y_normalized)
        # print('y =', y)
        return y_normalized
    
    def load_weights_from_file(self, filename='weights.txt'):
        with open(filename, 'r') as f:
            weights = json.loads(f.read())
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                neuron.weights = weights[i][j]
    
    def save_weights_to_file(self, filename='weights.json'):
        weights = []
        for layer in self.layers:
            layer_weights = []
            for neuron in layer.neurons:
                layer_weights.append(neuron.weights)
            weights.append(layer_weights)
        with open(filename, 'w') as f:
            f.write(json.dumps(weights))
    
    def save_normalization_parameters(self, filename='normalization_parameters.json'):
        with open(filename, 'w') as f:
            f.write(json.dumps(self.normalization_parameters))
            print('normalization_parameters =', self.normalization_parameters)
        
    def load_normalization_parameters(self, filename='normalization_parameters.json'):
        with open(filename, 'r') as f:
            self.normalization_parameters = json.load(f)
            print('normalization_parameters =', self.normalization_parameters)

    # def min_max_normalization(self, A, min_, max_):
    #     A = np.array(A)
    #     return ((A - min_) / (max_ - min_)).tolist()
    
    # def min_max_unnormalization(self, A, min_, max_):
    #     A = np.array(A)
    #     min_ = np.array(min_)
    #     max_ = np.array(max_)
    #     return (A * (max_ - min_) + min_).tolist()

    def normalize(self, A, means, stds):
        A = np.array(A)
        # means = np.array(means)
        # stds = np.array(stds)
        # return ((A) / stds).tolist()
        return ((A - means) / stds).tolist()
    
    def unnormalize(self, A, means, stds):
        A = np.array(A)
        means = np.array(means)
        stds = np.array(stds)
        # return (A * stds).tolist()
        return (A * stds + means).tolist()

    # def normalize_data(self, X, Y, w_means, w_stds, y_means):
    #     # normalize data
    #     X = np.array(X)
    #     Y = np.array(Y)
    #     x_means = X.mean(axis=0)
    #     y_means = Y.mean(axis=0)
    #     x_stds = X.std(axis=0)
    #     y_stds = Y.std(axis=0)
    #     X = (X - x_means) / x_stds
    #     Y = (Y - y_means) / y_stds

    #     self.normalization_parameters = {'w_means': w_means, 'y_means': y_means, 'w_stds': w_stds, 'y_stds': y_stds}
    #     return X.tolist(), Y.tolist()

if __name__ == '__main__':
    import csv
    import matplotlib.pyplot as plt

    USE_CSV = True

    def shuffle_data(X, Y):
        X = np.array(X)
        Y = np.array(Y)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices].tolist(), Y[indices].tolist()

    # preprocess X,Y data
    def preprocess_data(X, Y):
        dividers = [0 for _ in range(len(X[0]))]
        for x in X:
            for i, x_i in enumerate(x):
                dividers[i] = max(dividers[i], abs(x_i))
        for i, x in enumerate(X):
            for j, x_i in enumerate(x):
                X[i][j] = x_i / dividers[j]
        return X, Y
        

    def read_csv():
        X, Y = [], []
        with open('ce889_dataCollection.csv', 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        for row in data:
            X.append([float(row[0]), float(row[1])])
            Y.append([float(row[2]), float(row[3])]) 
        return X, Y


    if USE_CSV:
        # in this case the input layer will have 2 neurons and hidden layer will have 3 previous neurons
        X, Y = read_csv()
        #X, Y, w_means, y_means = normalize_data(X, Y)
        X, Y = shuffle_data(X, Y)
    else:
        # in this case the input layer will have 1 neuron and hidden layer will have 2 previous neurons
        X = [
            (2,)#,
            #(3,)  
        ]
        Y = [
            (3,2)#,
            #(4,3)
        ]

    # import pdb; pdb.set_trace()
    nn = NeuralNetHolder(learning_rate=0.01, momentum=0.1)
    # nn.load_weights_from_file(filename='weights.json')
    # import pdb; pdb.set_trace()
    costs = nn.train(deepcopy(X), deepcopy(Y), epochs=110) # 401

    with open('predictions.txt', 'w') as f:
        for x in X:
            # import pdb; pdb.set_trace()
            f.write(str(nn.predict(x, verbose=False)) + '\n')

    for x in X[:10]:
        print('x =', x, 'y =', nn.predict(x, verbose=False))

    nn.save_weights_to_file(filename='weights.json')
    nn.save_normalization_parameters(filename='normalization_parameters.json')
    # plot costs
    plt.plot(costs)
    plt.show()

    #print( nn.predict([1]) )