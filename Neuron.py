import random
import math

def sigmoid(x):
	# x = sum of activation value
	return 1 / (1 + math.exp(-x))

class Neuron:
	def __init__(self, activation_value=0, weights_count=2, is_bias=False, weights_values=None, activation_function=None):
		# weights_count is the number of neurons in previous layer
		self.activation_value = activation_value
		if weights_values is None:
			self.weights = [ random.uniform(0, 1) for _ in range(weights_count) ]
		else:
			self.weights = list(weights_values)
		self.is_bias = is_bias
		if is_bias:
			self.weights = []
		self.error = 0

		# Weights should be updated at the end of the backward propagation, 
		# otherwise their new values will affect calculations of previous weights
		self.new_weights = list(self.weights)

		# last weight change is used for momentum
		self.last_weight_changes = [0] * len(self.weights)

		self.activation_function = activation_function

	def activate(self, prev_layer):
		''' Calculate activation value of the neuron based on activation
		values of neurons in the previous layer and their corresponding weights. '''
		#import pdb; pdb.set_trace()
		sum_of_products = sum( prev_layer.neurons[i].activation_value * self.weights[i] for i in range(len(self.weights)) )
		
		#sum_of_products = sum( self.multiply_weights(prev_layer) )
		if self.activation_function is not None:
			self.activation_value = self.activation_function(sum_of_products)
		else:
			self.activation_value = sum_of_products

	def update_weights(self):
		''' Method used to update weights after backward propagation is finished.
		Weights couldn't be updated immediately because they are used to calculate
		errors of neurons in the previous layers. '''
		self.weights = list(self.new_weights)

	def multiply_weights(self, prev_layer):
		return [ neuron.activation_value * weight for neuron, weight in zip(prev_layer.neurons, self.weights) ]
