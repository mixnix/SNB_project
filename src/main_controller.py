import random

from src.neural_network_classifier import *
from matplotlib import pyplot as plt

from src.DataWorker import *

path = "../data/house_prices/data.csv"
dataWorker = DataWorker(path)
data = dataWorker.get_data()

#create classifier
prepared_train_vector = data[0:1300]
prepared_test_vector = data[1300:]

nn_classifier = NeuralNetworkClassifier(prepared_train_vector, prepared_test_vector, hidden_neurons=[5],
                                        learning_rates=[0.03], epochs=100,
                                        number_of_inputs=10, number_of_outputs=1)

# porownam kilka losowych przewidywan z faktycznymi cenami (z test set-u)
# nie wazne ze przyklady moga sie czasem powtarzac
random_examples = []
for i in range(4):
    random_examples.append(random.choice(prepared_test_vector))

# for now always uses first network in a classifier, will do it better after redesigning architecture of system
tablica_wynikow = nn_classifier.show_results_for_examples(random_examples)




train_error_matrix, test_error_matrix = nn_classifier.train_show_error()

#sprawdz accuracy na poczatku

#plot error


for i in range(0, len(train_error_matrix)):
    train_error_tuple = train_error_matrix[i]
    test_error_tuple = test_error_matrix[i]
    plt.figure()
    x_axis_values = range(0,len(train_error_tuple[0]))
    plt.plot(x_axis_values, train_error_tuple[0], 'r', x_axis_values, test_error_tuple[0], 'b')
    plt.title(train_error_tuple[1])
    plt.ylabel('train error')
    plt.xlabel('epoch')
    plt.show()


# for now always uses first network in a classifier, will do it better after redesigning architecture of system
tablica_wynikow2 = nn_classifier.show_results_for_examples(random_examples)

print("im done")