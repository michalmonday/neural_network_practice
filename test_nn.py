
#from Neuron import Neuron
from Layer import Layer

import math

LAMBDA_LEARNING_RATE = 0.8

def rounded_sigmoid(x):
	# Created just to make results more similar to provided example where
	# numbers were rounded on purpose for easier presentation on paper.
	# x = sum of activation value
	return round(1 / (1 + math.exp(-LAMBDA_LEARNING_RATE*x)), 2)

def sigmoid(x):
	# x = sum of activation value
	return 1 / (1 + math.exp(-LAMBDA_LEARNING_RATE*x))

layers = [ 
	Layer(prev_neurons_count=0, neurons_count=1, add_bias=True),
	Layer(prev_neurons_count=2, neurons_count=2, add_bias=True, neurons_weights_values=[
		[0.6, 0.7], # hidden layer, 1st node weights
		[0.8, 0.8]  # hidden layer, 2nd node weights
	]),
	Layer(prev_neurons_count=3, neurons_count=2, add_bias=False, neurons_weights_values=[
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


def train(X):
	for x_index, (x, y) in enumerate(zip(X, Y)):
		# set first layer activeation values to inputs to neural network
		for i, neuron in enumerate(layers[0].neurons[1:]): # omitting 1st because it's 1 by default (bias)
			#assert len(layers[0].neurons) == len(x), f"Length of first layer ({len(layers[0].neurons)}) doesn't match the length of first training example (x = {len(x)})."
			neuron.activation_value = x[i]
		# show_network(layers)

		for i, layer in enumerate(layers[1:]):
			# for each neuron in a layer, set activation value 
			# (based on previous weights and previous layer activation values)
			for j, neuron in enumerate(layer.neurons):
				# if neuron is a bias, we can assume it already has an activation value (1)
				# so we don'n need to set it based on previous neurons 
				if neuron.is_bias:
					continue
				sum_of_products = 0
				# for each neuron in previous layer
				for k, prev_neuron in enumerate(layers[i].neurons):
					# add product to the total sum
					print('prev_neuron.activation_value =', prev_neuron.activation_value, ',  neuron.weights[k] =', neuron.weights[k], ',  k =', k, f',  layer = {i+1}')
					#if i == LAST_LAYER_INDEX:
						#import pdb; pdb.set_trace()
					sum_of_products += prev_neuron.activation_value * neuron.weights[k]

				if i < LAST_LAYER_INDEX - 1:
					sum_of_products = sigmoid(sum_of_products)
				print (x_index, i, j, sum_of_products)
				neuron.activation_value = sum_of_products


		results = [ neuron.activation_value for neuron in layers[-1].neurons ]
		errors = [ o - r for r,o in zip(results, y) ]

		# set errors of the last layer before beginning backpropagation
		for i, (neuron, o) in enumerate(zip(layers[-1].neurons, y)):
			# r = result (prediction), o = observation
			layers[-1].neurons[i].error = o - neuron.activation_value

		# backpropagation (updating weights based on their prediction accuracy)
		# for each layer except input layer
		for i in range(len(layers)-1, 0, -1):
			activation_values = [neuron.activation_value for neuron in layers[i].neurons]
			for j, neuron in enumerate(layers[i].neurons):
				if i == LAST_LAYER_INDEX:
					neuron.error = y[j] - neuron.activation_value
				else:
					#import pdb; pdb.set_trace()
					neuron.error =  LAMBDA_LEARNING_RATE * neuron.activation_value * (1-neuron.activation_value) * sum(n.weights[j] * n.error for n in layers[i+1].neurons)
				
				for k, weight in enumerate(neuron.weights):
					neuron.new_weights[k] += LAMBDA_LEARNING_RATE * neuron.error * layers[i-1].neurons[k].activation_value 

		# Latch all neurons weights
		for i in range(len(layers)-1, 0, -1):
			for neuron in layers[i].neurons:
				neuron.latch_weights()

		# TODO: Fix backward propagation error.
		#       The reason of error is that the weights should be updated at the end of the backward
		#       propagation, otherwise their new values will affect calculations of previous weights...
		show_network(layers)

		print('results =', results)
		print('y =', y)
		print('errors =', errors)
		#print('total cost =', cost(results, y))


train(X)

