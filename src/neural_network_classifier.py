from src.neural_network import *


class NeuralNetworkClassifier:
    def __init__(self, train_data, test_data, hidden_neurons=[30], learning_rates=[0.3], epochs=10,
                 number_of_inputs=64, number_of_outputs=10):
        self.neural_networks_table = []
        for lrn_rate in learning_rates:
            for neurons_number in hidden_neurons:
                self.neural_networks_table.append(neural_network(neurons_number=neurons_number, learning_rate=lrn_rate,
                                                                 number_of_inputs=number_of_inputs,
                                                                 number_of_outputs=number_of_outputs))
        self.number_of_outputs = number_of_outputs
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs

    def prepare_example(self, example):
        slash_index = self.number_of_outputs
        return np.expand_dims(example[0:-slash_index], axis=1).T, np.expand_dims(example[-slash_index:], axis=1).T
        # output_vector = [0 for i in range(2)]
        # output_vector[int(example[-1])] = 1
        # output_vector = np.array([output_vector])
        # input_vector = np.array([example[0:-1]])
        # return input_vector, output_vector

    def train_show_error(self):
        train_error_matrix = []
        test_error_matrix = []
        for nn in self.neural_networks_table:
            train_error, test_error = self.train_one_network(nn)
            train_error_matrix.append(train_error)
            test_error_matrix.append(test_error)

        return train_error_matrix, test_error_matrix

    def train_one_network(self, network):
        train_error_vector = []
        test_error_vector = []
        # train for specified number of epochs
        for i in range(self.epochs):
            print("epoch " + str(i))
            # train network
            rand_permutated_data = np.random.permutation(self.train_data)
            for example in rand_permutated_data:
                input_vector, output_vector = self.prepare_example(example)
                network.train(input_vector, output_vector)

            # calculate error
            train_error = 0
            for example in self.train_data:
                input_vector, output_vector = self.prepare_example(example)
                train_error += network.squared_error(input_vector, output_vector)

            test_error = 0
            for example in self.test_data:
                input_vector, output_vector = self.prepare_example(example)
                test_error += network.squared_error(input_vector, output_vector)

            train_error_vector.append(train_error)
            test_error_vector.append(test_error)

        network_string = "learning rate: " + str(network.learning_rate) + " hidden neurons: " + str(network.hidden_neurons)
        return (train_error_vector, network_string), (test_error_vector, network_string)

    def all_networks_train_accuracy(self):
        accuracy_table = []
        for nn in self.neural_networks_table:
            accuracy_table.append(self.one_network_train_accuracy(nn))

        return accuracy_table

    def all_networks_test_accuracy(self):
        accuracy_table = []
        for nn in self.neural_networks_table:
            accuracy_table.append(self.one_network_test_accuracy(nn))

        return accuracy_table

    def one_network_train_accuracy(self, network):
        true_positives = 0
        for example in self.train_data:
            input_vector, output_vector = self.prepare_example(example)
            classification = network.classify(input_vector)
            if example[-1] == classification:
                true_positives += 1

        accuracy = true_positives / len(self.train_data)

        return (accuracy, "learning_rate: " +  str(network.learning_rate) + " hidden neurons: " + str(network.hidden_neurons))

    def one_network_test_accuracy(self, network):
        true_positives = 0
        for example in self.test_data:
            input_vector, output_vector = self.prepare_example(example)
            classification = network.classify(input_vector)
            if example[-1] == classification:
                true_positives += 1

        accuracy = true_positives / len(self.test_data)

        return (accuracy, "learning_rate: " + str(network.learning_rate) + " hidden neurons: " + str(network.hidden_neurons))

    # for now always uses first network in a classifier
    def show_results_for_examples(self, examles_vectors):
        results = []
        network = self.neural_networks_table[0]
        for vector in examles_vectors:
            input_v, output_v = self.prepare_example(vector)
            results.append((output_v, network.calculateValue(input_v)[0]))
        return results




