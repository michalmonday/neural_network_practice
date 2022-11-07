#from Neuron import Neuron
from Layer import Layer
import math

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
    return c 

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
        self.layers = [ 
            # input layer
            Layer(prev_neurons_count=0, neurons_count=1, add_bias=True), 

            # hidden layer
            Layer(prev_neurons_count=2, neurons_count=2, add_bias=True, activation_function=sigmoid, neurons_weights_values=[
                [0.6, 0.7], # hidden layer, 1st node weights
                [0.8, 0.8]  # hidden layer, 2nd node weights
            ]),

            # output layer
            Layer(prev_neurons_count=3, neurons_count=2, add_bias=False, activation_function=linear, neurons_weights_values=[
                [0.4, 0.5, 0.5], # output layer, 1st node weights
                [0.5, 0.7, 0.9]  # output layer, 2nd node weights
            ])
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
                neuron.update_weights() #neuron.weights = list(neuron.new_weights)


    def train(self, X, Y, epochs=1):

        costs = []
        # epoch is a single pass through the entire training dataset
        for epoch_index in range(epochs):
            epoch_cost = 0
            for i, (x, y) in enumerate(zip(X, Y)):
                results = self.forward_propagation(x)
                self.backward_propagation(y)
                #errors = [ o - r for r,o in zip(results, y) ]

                print('results =', results)
                # print('y =', y)
                #print('errors =', errors)
                epoch_cost += cost(results, y)
            costs.append(epoch_cost)
        return costs
    
    def predict(self, x):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        return self.forward_propagation(x)

if __name__ == '__main__':
    import csv
    import matplotlib.pyplot as plt

    USE_CSV = False
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
    else:
        # in this case the input layer will have 1 neuron and hidden layer will have 2 previous neurons
        X = [
            (2,),
            (3,)  
        ]
        Y = [
            (3,2),
            (4,3)
        ]

    # import pdb; pdb.set_trace()
    nn = NeuralNetHolder(learning_rate=0.1, momentum=0.1)
    costs = nn.train(X, Y, epochs=30)
    # plot costs
    plt.plot(costs)
    plt.show()

    print( nn.predict([1]) )