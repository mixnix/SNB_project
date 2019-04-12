from src.NeuralNetwork import *

class NNClassifier:
    # inicjalizacja podstawowych parametrow sieci neuronowych
    def __init__(self, train_data, test_data, hidden_neurons=30, learning_rates=0.3, epochs=100,
                 number_of_inputs=10, number_of_outputs=1):
        self.neural_network = neural_network(neurons_number=hidden_neurons, learning_rate=learning_rates,
                                                                 number_of_inputs=number_of_inputs,
                                                                 number_of_outputs=number_of_outputs)
        self.number_of_outputs = number_of_outputs
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs

    # odcina czesc wyjsciowa od wejsciowej
    def prepare_example(self, example):
        # just slashes vector to input vector and output vector
        slash_index = self.number_of_outputs
        return np.expand_dims(example[0:-slash_index], axis=1).T, np.expand_dims(example[-slash_index:], axis=1).T

    # trenuje siec i zwraca blad
    def train_network_get_error(self):
        # iterate over epochs
        train_error_vector = []
        test_error_vector = []

        for i in range(self.epochs):
            print("epoch " + str(i))
            rand_permutated_data = self.permutate_data(self.train_data)

            self.train_network_one_epoch(rand_permutated_data)

            train_error_vector.append(self.train_network_get_train_error())

            test_error_vector.append(self.train_network_get_test_error())

        return train_error_vector, test_error_vector

    # losowo miesza przyklady
    def permutate_data(self, data):
        # randomly permutates data
        return np.random.permutation(self.train_data)

    # jeden epoch trenowania sieci, bierze kazdy przyklad i na kazdym przekladzie trenuje
    def train_network_one_epoch(self, data):
        # trains network with every exampmle from data
        # doesnt permutate data before training
        for example in data:
            input_vector, output_vector = self.prepare_example(example)
            self.neural_network.train(input_vector, output_vector)

    # liczy blad dla sieci w danym momencie dla danych treningowych
    def train_network_get_train_error(self):
        # calculates error for whole training set and returns it as one number
            # squared_error
        train_error = 0
        for example in self.train_data:
            input_vector, output_vector = self.prepare_example(example)
            # train_error += self.neural_network.squared_error(input_vector, output_vector)
            train_error += abs(float(output_vector - self.neural_network.calculateValue(input_vector)[0]))

        return train_error / len(self.train_data)

    # liczy blad dla sieci w danym momencie dla danych testowych
    def train_network_get_test_error(self):
        test_error = 0
        for example in self.test_data:
            input_vector, output_vector = self.prepare_example(example)
            # test_error += self.neural_network.squared_error(input_vector, output_vector)
            test_error += abs(float(output_vector - self.neural_network.calculateValue(input_vector)[0]))
        # calculates test error for whole test set and returns it as one number
        return test_error / len(self.test_data)

    # liczy procentowy blad dla kazdego przykladu w train secie
    def get_final_train_percentage_error_matrix(self):
        final_train_error = []
        for example in self.train_data:
            input_vector, output_vector = self.prepare_example(example)
            final_train_error.append(abs(float(output_vector - self.neural_network.calculateValue(input_vector)[0]))/output_vector[0][0])
        return final_train_error

    # liczy procentowy blad dla kazdego przykladu w test secie
    def get_final_test_percentage_error_matrix(self):
        final_test_error = []
        for example in self.test_data:
            input_vector, output_vector = self.prepare_example(example)
            final_test_error.append(abs(float(output_vector - self.neural_network.calculateValue(input_vector)[0]))/output_vector[0][0])
        return final_test_error
