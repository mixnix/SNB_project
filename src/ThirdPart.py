from src.NNClassifier import *
from src.NNClassifierProxy import *
from src.DataWorker import *
from src.NNClassifiersMatrix import NNClassifiersMatrix

# ścieżka do danych
path = "../data/house_prices/data.csv"
# załądowanie danych i preprocessing
dataWorker = DataWorker(path)
# załądowanie danych do zmiennej
data = dataWorker.get_data()

# podzielenie danych na zbiór treningowy i zbiór testowy
prepared_train_vector = data[0:1300]
prepared_test_vector = data[1300:]

# stworzenie obiektu ktory bedzie przechowywal kilka obiektow
nn_classifiers_matrix = NNClassifiersMatrix(hidden_neurons =[3,6,12], learning_rates=[0.03, 0.1, 0.3],
                                            train_vector=prepared_train_vector, test_vector=prepared_test_vector)

# wytrenowanie kazdego klasyfikatora
nn_classifiers_matrix.train_all_classifiers()

# narysowanie sredniego bledu
nn_classifiers_matrix.paintAverageAccuracyGraph()

# narysowanie błędu prcoentwoego dla kazdego przykladu w training set
# narysowanie błędu prcoentwoego dla kazdego przykladu w test set
nn_classifiers_matrix.paintPercentageError()

