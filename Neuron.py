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
		sum_of_products = sum( prev_layer.neurons[i].activation_value * self.weights[i] for i in range(len(self.weights)) )
		if self.activation_function is not None:
			self.activation_value = self.activation_function(sum_of_products)
		else:
			self.activation_value = sum_of_products

	def update_weight(self):
		pass

	def multiply_weights(self, next_neuron_weight):
		# forward propagation to the next layer
		return self.activation_value * next_neuron_weight

	def forward_propagation(self, previous_neurons_value):
		pass
