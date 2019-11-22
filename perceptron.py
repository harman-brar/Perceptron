import numpy

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

training_inputs = numpy.array([[0,0,1],
                               [1,1,1],
                               [0,1,1],
                               [1,0,1]])

training_outputs = numpy.array([[0,1,0,1]]).T

numpy.random.seed(1)

synaptic_weights = 2 * numpy.random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

for iteration in range(500000):
    input_layer = training_inputs
    outputs = sigmoid(numpy.dot(input_layer, synaptic_weights))
    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)
    synaptic_weights += numpy.dot(input_layer.T, adjustments)

print('Synaptic weights after training: ')
print(synaptic_weights)

print('Outputs after training: ')
print(outputs)
