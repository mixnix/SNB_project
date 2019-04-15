from src.NNClassifier import *
from src.NNClassifierProxy import *
from src.DataWorker import *

# ścieżka do danych
path = "../data/house_prices/data.csv"
# załądowanie danych i preprocessing
dataWorker = DataWorker(path)
# załądowanie danych do zmiennej
data = dataWorker.get_data()

# podzielenie danych na zbiór treningowy i zbiór testowy
prepared_train_vector = data[0:1300]
prepared_test_vector = data[1300:]

# stworzenie klasyfikatora i załadowanie do niego danych
nn_classifier = NNClassifier(prepared_train_vector, prepared_test_vector)

# stworzenie proxy które będzie przechowywało wykresy i je rysowało
nn_proxy = NNClassifierProxy(nn_classifier)

# wytrenowanie sieci
nn_proxy.train()

# narysowanie średniego błędu
nn_proxy.paintAverageAccuracyGraph()

# narysowanie błędu prcoentwoego dla kazdego przykladu w training set
nn_proxy.paintTrainPercentageFinalError()

# narysowanie błędu prcoentwoego dla kazdego przykladu w test set
nn_proxy.paintTestPercentageFinalError()

# przewidź cenę 5 losowych nieruchomosci
nn_proxy.fiveEstateTest()

