from src.NNClassifier import NNClassifier
from matplotlib import pyplot as plt

class NNClassifiersMatrix:
    def __init__(self, hidden_neurons, learning_rates, train_vector, test_vector):
        self.NNClassifierTable = []
        for i in hidden_neurons:
            for j in learning_rates:
                self.NNClassifierTable.append(NNClassifier(train_vector, test_vector, hidden_neurons=i, learning_rates=j))

    def train_all_classifiers(self):
        self.error_table = []
        for nn_classifier in self.NNClassifierTable:
            temp_train_error, temp_test_error = nn_classifier.train_network_get_error()
            self.error_table.append((temp_train_error, temp_test_error))

    def paintAverageAccuracyGraph(self):
        for idx, error_couple in enumerate(self.error_table):

            plt.figure()
            x_axis_values = range(0, len(error_couple[0]))
            train_line, = plt.plot(x_axis_values, error_couple[0], 'r', label='train')
            test_line, = plt.plot(x_axis_values, error_couple[1], 'b', label='test')
            plt.legend(handles=[train_line, test_line])
            ll = self.NNClassifierTable[idx].neural_network.learning_rate
            hid_neu = self.NNClassifierTable[idx].neural_network.hidden_neurons
            plt.title('average accuracy ll: ' + str(ll) + ' hid_neu: ' + str(hid_neu))
            plt.ylabel('train error')
            plt.xlabel('epoch')
            plt.show()

    def paintPercentageError(self):
        for idx, classifier in enumerate(self.NNClassifierTable):
            final_percentage_train_error_matrix = classifier.get_final_train_percentage_error_matrix()
            final_percentage_test_error_matrix = classifier.get_final_test_percentage_error_matrix()

            final_percentage_train_error_matrix.sort()
            final_percentage_test_error_matrix.sort()

            plt.figure()
            x_axis_values = range(0, len(final_percentage_train_error_matrix))
            train_final_line, = plt.plot(x_axis_values, final_percentage_train_error_matrix, 'r',
                                         label='train_final')
            plt.legend(handles=[train_final_line])
            ll = classifier.neural_network.learning_rate
            hid_neu = classifier.neural_network.hidden_neurons
            plt.title('blad treningowy dla kazdego przykladu ll: ' + str(ll) + ' hid_neu: ' + str(hid_neu))
            plt.ylabel('procent')
            plt.xlabel('przyklad')
            plt.show()

            plt.figure()
            x_axis_values = range(0, len(final_percentage_test_error_matrix))
            train_final_line, = plt.plot(x_axis_values, final_percentage_test_error_matrix, 'r',
                                         label='train_final')
            plt.legend(handles=[train_final_line])
            ll = classifier.neural_network.learning_rate
            hid_neu = classifier.neural_network.hidden_neurons
            plt.title('blad testowy dla kazdego przykladu ll: ' + str(ll) + ' hid_neu: ' + str(hid_neu))
            plt.ylabel('procent')
            plt.xlabel('przyklad')
            plt.show()