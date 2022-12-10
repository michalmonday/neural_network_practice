# class Person:
#     def __init__(self):
#         self.name = 'John'
#         self.age = 25
#         self.height = 180
    
#     def go_library(self):
#         if self.age > 30: 
#             self.walk_slowly(20)
#         else:
#             self.walk_fast(10)

#     def walk_slowly(self, n):
#         print('walking slowly')
    
#     def walk_fast(self, n):
#         print('walking fast')

#     def print_self(self):
#         print(self)
# p = Person()

# print(p)
# print(p.print_self())

# exit()


#from Neuron import Neuron
from Layer import Layer
import json
from copy import deepcopy
import random
import numpy as np

def sigmoid(sum_of_products, lambda_=1):
    sum_of_products = min(500, sum_of_products)
    sum_of_products = max(-500, sum_of_products)
    return 1 / (1 + np.exp(-lambda_*sum_of_products))

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
    ''' rmse'''
    c = 0
    for r, o in zip(results, observations):
        try:
            c += (o - r) **2 
        except OverflowError:
            print('overflow when calculating cost, r={}, o={}'.format(r, o))            
    return np.sqrt(c / len(results))

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
           #Layer(prev_neurons_count=3, neurons_count=33, add_bias=True, activation_function=sigmoid),
        #    Layer(prev_neurons_count=3, neurons_count=24, add_bias=True, activation_function=tanh),
           Layer(prev_neurons_count=3, neurons_count=34, add_bias=True, activation_function=sigmoid),
        #    Layer(prev_neurons_count=3, neurons_count=12, add_bias=True, activation_function=tanh),

           # output layer
        #    Layer(prev_neurons_count=25, neurons_count=2, add_bias=False, activation_function=linear)
           Layer(prev_neurons_count=35, neurons_count=2, add_bias=False, activation_function=linear)
        #    Layer(prev_neurons_count=13, neurons_count=2, add_bias=False, activation_function=linear)
           ]
        self.LAST_LAYER_INDEX = len(self.layers) - 1


        # load keras model from filename
        #self.model = tf.keras.models.load_model('model.h5')    


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
            neuron.error = o - neuron.activation_value

        # backpropagation (updating weights based on their prediction accuracy)
        # for each layer except input layer
        for i in range(len(self.layers)-1, 0, -1):
            for j, neuron in enumerate(self.layers[i].neurons):
                # last layer error is measured by the difference between prediction and observation
                # for other layers, error is measured using the sum of errors (multiplied by weights) of neurons in the next layer
                if i == self.LAST_LAYER_INDEX:
                    neuron.error = y[j] - neuron.activation_value 
                    # neuron.error = neuron.activation_value - y[j] 
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
                neuron.update_weights(epsilon_regularisation=0.1) #neuron.weights = list(neuron.new_weights)

    def compute_means(self, A):
        return [ sum(a) / len(a) for a in zip(*A) ]
    
    def compute_standard_deviations(self, A, means):
        return [ np.sqrt(sum((a[i] - means[i])**2 for a in A) / len(A)) for i in range(len(A[0])) ]

    def train(self, X, Y, validation_X=[], validation_Y=[], epochs=1):
        # x_mins = np.min(X, axis=0)
        # x_maxs = np.max(X, axis=0)
        # y_mins = np.min(Y, axis=0)
        # y_maxs = np.max(Y, axis=0)

        x_means = self.compute_means(X)
        x_stds = self.compute_standard_deviations(X, x_means)
        y_means = self.compute_means(Y)
        y_stds = self.compute_standard_deviations(Y, y_means)

        # x_means_np = np.mean(X, axis=0)
        # x_stds_np = np.std(X, axis=0)
        # y_means_np = np.mean(Y, axis=0)
        # y_stds_np = np.std(Y, axis=0)

        X_orig = deepcopy(X)
        Y_orig = deepcopy(Y)
        X = self.normalize(X, x_means, x_stds)
        Y = self.normalize(Y, y_means, y_stds)

        validation_X_orig = deepcopy(X)
        validation_Y_orig = deepcopy(Y)
        on_X = self.normalize(validation_X, x_means, x_stds)
        validation_Y = self.normalize(validation_Y, y_means, y_stds)
        # self.normalization_parameters = {'x_means': x_means.tolist(), 'x_stds': x_stds.tolist(), 'y_means': y_means.tolist(), 'y_stds': y_stds.tolist()}
        self.normalization_parameters = {'x_means': x_means, 'x_stds': x_stds, 'y_means': y_means, 'y_stds': y_stds}
        costs = []
        validation_costs = []
        learning_rate_decreases = []
        # epoch is a single pass through the entire training dataset
        for epoch_index in range(epochs):
            epoch_cost = 0
            validation_epoch_cost = 0

            # calculate validation data cost
            for i, (x, y, y_orig) in enumerate(zip(validation_X, validation_Y, validation_Y_orig)):
                y_pred = self.forward_propagation(x)
                results = self.unnormalize([y_pred], y_means, y_stds)[0]
                validation_epoch_cost += abs(cost(results, y_orig) / len(validation_X))
            if epoch_index > 1 and validation_epoch_cost > validation_costs[-1]:
                print('Early stopping because validation cost is increasing')
                # return costs, validation_costs, learning_rate_decreases
            validation_costs.append(validation_epoch_cost)

            for i, (x, y, y_orig) in enumerate(zip(X, Y, Y_orig)):
                y_pred = self.forward_propagation(x)
                results = self.unnormalize([y_pred], y_means, y_stds)[0]
                # errors = [ o - r for r,o in zip(results, y) ]
                self.backward_propagation(y)


                # print('results =', results)
                # print('y =', y)
                # print('errors =', errors)
                # print('self.predict(x) =', self.predict(X_orig[i]))
                try:
                    epoch_cost += abs(cost(results, y_orig) / len(X))
                except OverflowError:
                    print(f'OverflowError, results = {results}, y = {y_orig}')
                    
                # print('cost =', cost(results, y_orig))
            if epoch_index > 1 and epoch_cost > costs[-1]:
                self.learning_rate *= 0.8
                learning_rate_decreases.append(1)
            else:
                learning_rate_decreases.append(0)
            costs.append(epoch_cost)

            show_network(self.layers)
            print()
            print(f'{epoch_index+1}. cost = {epoch_cost}, validation cost = {validation_epoch_cost}')
        return costs, validation_costs, learning_rate_decreases

    def predict(self, x, verbose=True):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        # [
        #  x_dist,  - high value means goal is on the right
        #  y_dist,  - high value means goal is below
        #  x_speed  - high value means spaceship is moving right
        #  y_speed, - high value means spaceship is moving down
        # ]
        if type(x) == str:
            x = [float(x_) for x_ in x.split(',')]
            x[0] -= 20.0
            print('x = ', x)
        x_normalized = self.normalize([x], self.normalization_parameters['x_means'], self.normalization_parameters['x_stds'])[0]
        y_normalized = self.forward_propagation(x_normalized)
        y = self.unnormalize([y_normalized], self.normalization_parameters['y_means'], self.normalization_parameters['y_stds'])[0]
        if verbose:
            print('x =', x, 'x_normalized =', x_normalized)
            print('y =', y, 'y_normalized =', y_normalized)
        return y
    
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

    def normalize(self, A, means, stds):
        A_norm = []
        for a in A:
            a_norm = [(a[i] - means[i]) / stds[i] for i in range(len(a))]
            A_norm.append(a_norm)
        return A_norm
    
    def unnormalize(self, A_norm, means, stds):
        A = []
        for a in A_norm:
            a = [(a[i] * stds[i]) + means[i] for i in range(len(a))]
            A.append(a)
        return A


if __name__ == '__main__':
    import csv
    import matplotlib.pyplot as plt


    USE_CSV = True

    def shuffle_data(X, Y):
        X_shuffled = []
        Y_shuffled = []
        indices = list(range(len(X)))
        random.shuffle(indices)
        for i in indices:
            X_shuffled.append(X[i])
            Y_shuffled.append(Y[i])
        return X_shuffled, Y_shuffled

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

    def clean_data(X, Y):
        new_X = []
        new_Y = []
        for i, (x, y) in enumerate(zip(X, Y)):
            # positive x[0] means spaceship is on the left from the goal
            # positive x[1] means spaceship is above the goal (it's always above)

            # positive y[0] means spaceship is moving right
            # positive y[1] means spaceship is moving down

            x[0] -= 20.0

            if x[0] < 0 and y[0] > 0:
                continue
            if x[0] > 0 and y[0] < 0:
                continue

            # don't let it move down when it's far away (remove these datapoints)
            if abs(x[0]) > 50 and y[1] > 0:
                continue
            new_X.append(x)
            new_Y.append(y) 
        return new_X, new_Y

        

    def read_csv():
        X, Y = [], []
        with open('ce889_dataCollection.csv', 'r') as f:
        # with open('ce889_dataCollection_full.csv', 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        for row in data:
            X.append([float(row[0]), float(row[1])])
            Y.append([float(row[2]), float(row[3])]) 
        return X, Y


    if USE_CSV:
        # in this case the input layer will have 2 neurons and hidden layer will have 3 previous neurons
        X, Y = read_csv()
        X, Y = shuffle_data(X, Y)
        X, Y = clean_data(X, Y)
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

    # split data into training and validation sets
    validation_split = 0.15
    validation_index = int(len(X) * validation_split)
    validation_X = X[:validation_index]
    validation_Y = Y[:validation_index]
    X = X[validation_index:]
    Y = Y[validation_index:]

    nn = NeuralNetHolder(learning_rate=0.04, momentum=0.4)

    # nn.load_weights_from_file(filename='weights.json')
    costs, validation_costs, learning_rate_decreases = nn.train(
        deepcopy(X),
        deepcopy(Y),
        validation_X=validation_X,
        validation_Y=validation_Y,
        epochs=50
        #epochs=100
        ) # 401

    with open('predictions.txt', 'w') as f:
        for x in X:
            f.write(str(nn.predict(x, verbose=False)) + '\n')

    for x in X[:3]:
        print('x =', x, 'y =', nn.predict(x, verbose=False))

    nn.save_weights_to_file(filename='weights.json')
    nn.save_normalization_parameters(filename='normalization_parameters.json')
    # plot costs
    plt.plot(costs, label='training costs')
    plt.plot(validation_costs, label='validation costs')
    plt.plot(learning_rate_decreases, label='learning rate decreases')
    plt.legend()
    plt.show()

