import numpy as np

class DataWorker:

    def __init__(self, path, loadFresh = True):
        # last column is output

        # przyporzadkuj path do zmiennej
        # jesli load fresh jest False to zaladuj dane juz przetworzone z pliku do
        # ktorego je zapisales
        self.path = path
        self.load_data_from_file()
        self.categorize_data()
        self.scale_data()

    def load_data_from_file(self):
        # loads raw data from file
        self.data = np.loadtxt(open(self.path, "rb"), dtype=str, delimiter=",", skiprows=1)

    def categorize_data(self):
        # picks some categories of data and
        # converts them form string to numbers
        # Extracting columns that we will use
        # So we will use 9 input values (date will be one value)
        neighborhoodType = self.data[:, 12:13]
        bldgType = self.data[:, 15:16]
        houseStyle = self.data[:, 16:17]
        overallQual = self.data[:, 17:18]
        overalCond = self.data[:, 18:19]
        yearBuilt = self.data[:, 19:20]
        yearRemodAdd = self.data[:, 20:21]
        grLivArea = self.data[:, 46:47]
        monthSold = self.data[:, 76:77]
        yearSold = self.data[:, 77:78]

        targetPrice = self.data[:, 80:81].astype(float)

        neighborhoodType = self.categorize_strings(neighborhoodType, self.create_dict_from_string_list(neighborhoodType))
        bldgType = self.categorize_strings(bldgType, self.create_dict_from_string_list(bldgType))
        houseStyle = self.categorize_strings(houseStyle, self.create_dict_from_string_list(houseStyle))
        neighborhoodType = self.categorize_strings(neighborhoodType, self.create_dict_from_string_list(neighborhoodType))
        overallQual = overallQual.astype(float)
        overalCond = overalCond.astype(float)
        yearBuilt = yearBuilt.astype(float)
        yearRemodAdd = yearRemodAdd.astype(float)
        grLivArea = grLivArea.astype(float)
        monthSold = np.array([a[0] if len(a[0]) == 2 else "0" + a[0] for a in monthSold])
        dateSold = np.array([monthSold[i] + yearSold[i][0] for i in range(len(yearSold))]).astype(float)
        dateSold = dateSold[np.newaxis].T

        # create vectors of input + result(output)
        data_vector = np.concatenate(
            (neighborhoodType, bldgType, houseStyle, neighborhoodType, overallQual, overalCond, yearBuilt,
             yearRemodAdd, grLivArea, dateSold, targetPrice), axis=1)

        self.data = data_vector

    def categorize_strings(self, values_list, conversion_dictionary):
        # Not sure if it catches these values correctly
        if 'NA' in values_list:
            print("Tablica zawiera puste wartości")
        for idx, val in enumerate(values_list):
            values_list[idx] = conversion_dictionary[values_list[idx][0]]
        return values_list.astype(float)

    def create_dict_from_string_list(self, values):
        # convert list to set of unique tuples
        unique_set = set(map(tuple, values))
        # convert set to set of unique values
        unique_set = map(lambda x: x[0], unique_set)
        # convert set to list of unique values
        unique_list = list(unique_set)
        # create a dict with values from 1 to n where n is length of list
        return dict((unique_list[i - 1], i) for i in range(1, len(unique_list) + 1))

    def scale_data(self):
        scaled_input = self.scale_input()
        scaled_output = self.scale_output()
        self.data = np.append(scaled_input, scaled_output, axis=1)

    def scale_input(self):
        # normalizes input
        input_part = self.data[:, :-1]
        input_part = self.standarize(input_part)
        return input_part

    def scale_output(self):
        # normalizes output
        output_part = self.data[:, -1:]
        output_part = self.scale_target_function(output_part)
        return output_part

    def scale_target_function(self, train_results):
        return np.divide(train_results, max(train_results))


    def standarize(self, x):
        x -= np.mean(x, axis=0)
        x /= np.std(x, axis=0)
        x = np.nan_to_num(x)
        return x

    def get_data(self):
        # returns list of vectors
        # each vector contains input (beginning fields) and output (fields at the end)
        return self.data

