
#from Neuron import Neuron
from Layer import Layer

import math

LAMBDA_LEARNING_RATE = 0.8
MOMENTUM = 0.1

def rounded_sigmoid(x):
    # Created just to make results more similar to provided example where
    # numbers were rounded on purpose for easier presentation on paper.
    # x = sum of activation value
    return round(1 / (1 + math.exp(-LAMBDA_LEARNING_RATE*x)), 2)

def sigmoid(x):
    # x = sum of activation value
    return 1 / (1 + math.exp(-LAMBDA_LEARNING_RATE*x))

def linear(x):
    return x

layers = [ 
    # input layer
    Layer(prev_neurons_count=0, neurons_count=1, add_bias=True), 

    # hidden layer
    Layer(prev_neurons_count=2, neurons_count=2, add_bias=True, activation_function=rounded_sigmoid, neurons_weights_values=[
        [0.6, 0.7], # hidden layer, 1st node weights
        [0.8, 0.8]  # hidden layer, 2nd node weights
    ]),

    # output layer
    Layer(prev_neurons_count=3, neurons_count=2, add_bias=False, activation_function=linear, neurons_weights_values=[
        [0.4, 0.5, 0.5], # output layer, 1st node weights
        [0.5, 0.9, 0.7]  # output layer, 2nd node weights
    ]),
    ]

LAST_LAYER_INDEX = len(layers) - 1

print(layers)

X = [
    (2,)
    #(3,)

    #(346.56857844293313,307.1),
    #(346.56857844293313,307.1),
    #(346.52857844293317,307.20000000000005),
    #(346.52857844293317,307.40000000000003)
]

Y = [
    (3,2)
    #(4,3)

    # (0.0,0.0),
    # (-0.1,0.04000000000000001),
    # (-0.2,0.0),
    # (-0.30000000000000004,0.04000000000000001)
]

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
    # results are predicted values
    c = 0
    for r, o in zip(results, observations):
        c += o - r
    return c

def forward_propagation(layers, x):
    # set first layer activation values to inputs to neural network
    for i, neuron in enumerate(layers[0].neurons[1:]): # omitting 1st because it's 1 by default (bias)
        neuron.activation_value = x[i]

    for i, layer in enumerate(layers[1:]):
        # for each neuron in a layer, set activation value 
        # (based on previous weights and previous layer activation values)
        for j, neuron in enumerate(layer.neurons):
            # if neuron is a bias, we can assume it already has an activation value (1)
            # so we don'n need to set it based on previous neurons 
            if neuron.is_bias:
                continue
            # i in this case is the index of previous layer (because we're iterating over layers[1:])
            neuron.activate(layers[i])

    # activation values of the last layer are the results of the neural network
    results = [ neuron.activation_value for neuron in layers[-1].neurons ]
    return results

def backward_propagation(layers, y):
    # set errors of the last layer before beginning backpropagation
    for i, (neuron, o) in enumerate(zip(layers[-1].neurons, y)):
        # r = result (prediction), o = observation
        layers[-1].neurons[i].error = o - neuron.activation_value

    # backpropagation (updating weights based on their prediction accuracy)
    # for each layer except input layer
    for i in range(len(layers)-1, 0, -1):
        for j, neuron in enumerate(layers[i].neurons):
            # last layer error is measured by the difference between prediction and observation
            # for other layers, error is measured using the sum of errors (multiplied by weights) of neurons in the next layer
            if i == LAST_LAYER_INDEX:
                neuron.error = y[j] - neuron.activation_value
            else:
                neuron.error =  LAMBDA_LEARNING_RATE * neuron.activation_value * (1-neuron.activation_value) * sum(n.weights[j] * n.error for n in layers[i+1].neurons)
            
            for k, weight in enumerate(neuron.weights):
                weight_change = LAMBDA_LEARNING_RATE * neuron.error * layers[i-1].neurons[k].activation_value + MOMENTUM * neuron.last_weight_changes[k]
                # last weight changes are used to calculate momentum
                neuron.last_weight_changes[k] = weight_change
                # weights are not updated immediately because we need to use them to calculate errors of neurons in the previous layers
                neuron.new_weights[k] += weight_change

    # Latch all neurons weights
    for i in range(len(layers)-1, 0, -1):
        for neuron in layers[i].neurons:
            neuron.weights = list(neuron.new_weights)

def train(X, epochs=1):
    for epoch in range(epochs):
        for x_index, (x, y) in enumerate(zip(X, Y)):
            results = forward_propagation(layers, x)
            backward_propagation(layers, y)
            #errors = [ o - r for r,o in zip(results, y) ]

            show_network(layers)
            print('results =', results)
            # print('y =', y)
            #print('errors =', errors)
            #print('total cost =', cost(results, y))

train(X, epochs=1)

