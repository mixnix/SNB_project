import numpy as np


class neural_network:
    # zainicjowanie sieci neuronowej losowymi wartościami
    def __init__(self, neurons_number=30, learning_rate=0.3, number_of_inputs=64, number_of_outputs=10):
        self.W1 = np.random.uniform(low=0-1, high=1, size=(number_of_inputs, neurons_number))  # first is y and then x
        self.B1 = np.random.uniform(low=0-1, high=1, size=(1, neurons_number))
        self.W2 = np.random.uniform(low=0-1, high=1, size=(neurons_number, number_of_outputs))
        self.B2 = np.random.uniform(low=0-1, high=1, size=(1, number_of_outputs))

        self.hidden_neurons = neurons_number
        self.learning_rate = learning_rate

    # funkcja drukująca wartości wag w celach diagnostycznych
    def __str__(self):
        return ("W1 = \n" + str(self.W1) +
              "\nB1 = \n" + str(self.B1) +
              "\nW2 = \n" + str(self.W2) +
              "\nB2 = \n" + str(self.B2))

    # funkcja sigmoidalna aktywacji
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # funkcja tanh do celów testowych
    def tanh(self, x):
        return np.tanh(x)

    # pochodna funkcji sigmoidalnej
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # pochodna funkcji tanh
    def tanh_derivative(self, x):
        return (1 - (x ** 2))

    def relu_derivative(self, x):
        x[x<0] = 0
        x[x>0] = 1
        return x

    def relu(self, x):
        return np.maximum(x, 0)

    # w tej funkcji liczymy wartość wyjściową sieci nerunowej
    def calculateValue(self, inputVector):
        # mnożymy wejście przez wagi, dodajemy bias i przepuszczamy przez funkcje aktywacji
        firstLayerNet = np.dot(inputVector, self.W1) + self.B1
        firstLayerValue = self.sigmoid(firstLayerNet)
        # potem robimy to samo dla drugiej warstwy
        # mnożymy wyjście pierwszej warstwy przez wagi drugiej warstwy, dodajemy biad i
        # przepuszczamy prezz funkcję aktywacji
        secondLayerNet = np.dot(firstLayerValue, self.W2) + self.B2
        output = self.sigmoid(secondLayerNet)
        return output, firstLayerValue

    # funkcja aktualizująca wartości wag
    def train(self, inputVector, outputVector):

        # liczymy wartosć wyjściową na sieci i na pierwszej warstwie
        layer2, layer1 = self.calculateValue(inputVector)

        # liczymy błąd na wyjściu i o ile powinniśmy zmienić wagi drugiej warstwy
        layer2 = np.array(layer2)
        layer1 = np.array(layer1)
        layer2_error = outputVector - layer2
        layer2_delta = layer2_error*self.sigmoid_derivative(layer2)

        # korzystając z wartości delty na wyjściu liczymy o ile powinna się zmienić
        # wartość wag na pierwszej warstwie
        layer1_error = layer2_delta.dot(self.W2.T)
        layer1_delta = layer1_error *self.sigmoid_derivative(layer1)

        self.W2 += layer1.T.dot(layer2_delta) * self.learning_rate
        self.B2 = self.learning_rate * np.sum(layer2_delta, axis=0, keepdims=True)
        self.W1 += inputVector.T.dot(layer1_delta) * self.learning_rate
        self.B1 = self.learning_rate * np.sum(layer1_delta, axis=0, keepdims=True)
