import numpy as np
from matplotlib import pyplot as plt

class NNClassifierProxy:
    def __init__(self, nnClassifier, outScaleOrder):
        # stworz referencje do classifiera
        self.nnClassifier = nnClassifier
        self.outScaleOrder = outScaleOrder

    def train(self):
        # wywolaj metode trenujacac siec
        self.train_error_matrix, self.test_error_matrix = self.nnClassifier.train_network_get_error()

    def paintAverageAccuracyGraph(self):
        # graf bledu w kazdym epochu
        # i z powrotem wyskalowanie ceny do normalnych wartosci by
        # latwiej mozna bylo zobaczyc o ile sie myli siec (maksymalna wartosc ceny to 750 000
        self.train_error_matrix = np.multiply(self.train_error_matrix, self.outScaleOrder)
        self.test_error_matrix = np.multiply(self.test_error_matrix, self.outScaleOrder)

        plt.figure()
        x_axis_values = range(0, len(self.train_error_matrix))
        train_line, = plt.plot(x_axis_values, self.train_error_matrix, 'r', label='train')
        test_line, = plt.plot(x_axis_values, self.test_error_matrix, 'b', label='test')
        plt.legend(handles=[train_line, test_line])
        plt.title('average accuracy')
        plt.ylabel('train error')
        plt.xlabel('epoch')
        plt.show()

    # rysuje procentowy blad dla kazdego przykladu w train secie
    def paintTrainPercentageFinalError(self):
        self.final_percentage_train_error_matrix = self.nnClassifier.get_final_train_percentage_error_matrix()

        self.final_percentage_train_error_matrix.sort()

        plt.figure()
        x_axis_values = range(0, len(self.final_percentage_train_error_matrix))
        train_final_line, = plt.plot(x_axis_values, self.final_percentage_train_error_matrix, 'r', label='train_final')
        plt.legend(handles=[train_final_line])
        plt.title('blad dla kazdego przykladu')
        plt.ylabel('procent')
        plt.xlabel('przyklad')
        plt.show()

    # rysuje procentowy blad dla kazdego przykladu w test secie
    def paintTestPercentageFinalError(self):
        self.final_percentage_test_error_matrix = self.nnClassifier.get_final_test_percentage_error_matrix()

        self.final_percentage_test_error_matrix.sort()

        plt.figure()
        x_axis_values = range(0, len(self.final_percentage_test_error_matrix))
        test_final_line, = plt.plot(x_axis_values, self.final_percentage_test_error_matrix, 'r', label='train_final')
        plt.legend(handles=[test_final_line])
        plt.title('blad dla kazdego przykladu')
        plt.ylabel('procent')
        plt.xlabel('przyklad')
        plt.show()

    def fiveEstateTest(self):
        for i in range(1, 5):
            [real_value, calulated_value] = self.nnClassifier.calculateRandomEstate()
            real_value *= self.outScaleOrder
            calulated_value *= self.outScaleOrder
            print("Nieruchomosc: " + str(i) + " prawidłowa: " + str(real_value) + " przewidziana: " + str(
                calulated_value))
