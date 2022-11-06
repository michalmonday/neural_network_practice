from Neuron import Neuron


class Layer:
	
	def __init__(self, prev_neurons_count=0, neurons_count=0, add_bias=False, activation_function=None, neurons_weights_values=None):
		self.neurons = [ Neuron(weights_count=prev_neurons_count, weights_values=(neurons_weights_values[i] if neurons_weights_values is not None else None), activation_function=activation_function) for i in range(neurons_count) ]

		if add_bias:
			self.neurons.insert(0, Neuron(1, weights_count=prev_neurons_count, is_bias=True))

		