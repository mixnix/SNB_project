import numpy as np


class neural_network:
    def __init__(self, neurons_number=30, learning_rate=0.3, number_of_inputs=64, number_of_outputs=10):
        self.W1 = np.random.uniform(low=0-1, high=1, size=(number_of_inputs, neurons_number))  # first is y and then x
        self.B1 = np.random.uniform(low=0-1, high=1, size=(1, neurons_number))
        self.W2 = np.random.uniform(low=0-1, high=1, size=(neurons_number, number_of_outputs))
        self.B2 = np.random.uniform(low=0-1, high=1, size=(1, number_of_outputs))

        self.hidden_neurons = neurons_number
        self.learning_rate = learning_rate

    def __str__(self):
        return ("W1 = \n" + str(self.W1) +
              "\nB1 = \n" + str(self.B1) +
              "\nW2 = \n" + str(self.W2) +
              "\nB2 = \n" + str(self.B2))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh_derivative(self, x):
        return (1 - (x ** 2))

    def calculateValue(self, inputVector):
        firstLayerNet = np.dot(inputVector, self.W1) + self.B1
        #here we want tang because it has better gradient properties, it is centered around 0
        firstLayerValue = self.sigmoid(firstLayerNet)
        secondLayerNet = np.dot(firstLayerValue, self.W2) + self.B2
        #here we want probability so we want value from range [0, 1]
        output = self.sigmoid(secondLayerNet)
        return output, firstLayerValue

    def train(self, inputVector, outputVector):

        layer2, layer1 = self.calculateValue(inputVector)


        layer2 = np.array(layer2)
        layer1 = np.array(layer1)
        layer2_error = outputVector - layer2
        layer2_delta = layer2_error*self.sigmoid_derivative(layer2)

        layer1_error = layer2_delta.dot(self.W2.T)
        layer1_delta = layer1_error *self.sigmoid_derivative(layer1)

        self.W2 += layer1.T.dot(layer2_delta) * self.learning_rate
        self.W1 += inputVector.T.dot(layer1_delta) * self.learning_rate

    def classify(self, inputVector):
        values, values2 = self.calculateValue(inputVector)
        return values[0][0]

    def squared_error(self, inputVector, outputVector):
        layer2, layer1 = self.calculateValue(inputVector)
        layer2 = np.array(layer2)
        layer2_error = outputVector - layer2
        return np.sum(np.array(layer2_error)**2)
