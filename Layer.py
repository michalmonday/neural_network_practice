from Neuron import Neuron


class Layer:
	
	def __init__(self, prev_neurons_count=0, neurons_count=0, add_bias=False, neurons_weights_values=None):
		if neurons_weights_values is not None:
			self.neurons = [ Neuron(weights_count=prev_neurons_count, weights_values=neurons_weights_values[i]) for i in range(neurons_count) ]
		else:
			self.neurons = [ Neuron(weights_count=prev_neurons_count) for _ in range(neurons_count) ]


		if add_bias:
			self.neurons.insert(0, Neuron(1, weights_count=prev_neurons_count, is_bias=True))