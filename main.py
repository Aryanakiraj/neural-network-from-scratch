import numpy as np 

class Layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, matrix_inputs):
        self.output = np.maximum(0, matrix_inputs)
    
neurons = np.random.randint(1, 10, size=(3))
test_inputs = np.random.randint(1, 10, size=(3))

layer1 = Layer_dense(len(test_inputs), len(neurons))
layer1.forward(test_inputs)

print(layer1.output)
print()

activate = Activation_ReLU()
activate.forward(layer1.output)

print(activate.output)