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



