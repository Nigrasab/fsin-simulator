import numpy as np
from src.neuron import SfiralNeuron

class SfiralNetwork:
    def __init__(self, input_size, layer_size):
        self.neurons = [SfiralNeuron() for _ in range(layer_size)]
        # Инициализация весов с учетом принципа золотого вурфа (1.309)
        self.weights = np.random.uniform(-1, 1, (input_size, layer_size)) * 1.309

    def forward(self, inputs):
        outputs = []
        for i, neuron in enumerate(self.neurons):
            neuron_weights = self.weights[:, i]
            output = neuron.activate(inputs, neuron_weights)
            outputs.append(output)
        return np.array(outputs)
