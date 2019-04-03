import numpy as np


def standarize(x):
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)
    x = np.nan_to_num(x)
    return x


def scale_target_function(train_results, test_results):
    divide_index = len(train_results)
    results = np.append(train_results, test_results, axis=0)
    divider = max(results)
    results = np.divide(results, divider)
    return (results[0:divide_index], results[divide_index:], divider)


def create_dict_from_string_list(values):
    # convert list to set of unique tuples
    unique_set = set(map(tuple, values))
    # convert set to set of unique values
    unique_set = map(lambda x: x[0], unique_set)
    # convert set to list of unique values
    unique_list = list(unique_set)
    # create a dict with values from 1 to n where n is length of list
    return dict((unique_list[i - 1], i) for i in range(1, len(unique_list) + 1))


def categorize_strings(values_list, conversion_dictionary):
    # Not sure if it catches these values correctly
    if 'NA' in values_list:
        print("Tablica zawiera puste wartości")
    for idx, val in enumerate(values_list):
        values_list[idx] = conversion_dictionary[values_list[idx][0]]
    return values_list.astype(float)


def load_preprocess_house_data(path):
    # df = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/autos/imports-85.data",
    #                  header=None, na_values="?")
    # df.head()
    data = np.loadtxt(open(path, "rb"), dtype=str, delimiter=",", skiprows=1)

    # Extracting columns that we will use
    # So we will use 9 input values (date will be one value)
    neighborhoodType = data[:, 12:13]
    bldgType = data[:, 15:16]
    houseStyle = data[:, 16:17]
    overallQual = data[:, 17:18]
    overalCond = data[:, 18:19]
    yearBuilt = data[:, 19:20]
    yearRemodAdd = data[:, 20:21]
    grLivArea = data[:, 46:47]
    monthSold = data[:, 76:77]
    yearSold = data[:, 77:78]

    targetPrice = data[:, 80:81].astype(float)

    neighborhoodType = categorize_strings(neighborhoodType, create_dict_from_string_list(neighborhoodType))
    bldgType = categorize_strings(bldgType, create_dict_from_string_list(bldgType))
    houseStyle = categorize_strings(houseStyle, create_dict_from_string_list(houseStyle))
    neighborhoodType = categorize_strings(neighborhoodType, create_dict_from_string_list(neighborhoodType))
    overallQual = overallQual.astype(float)
    overalCond = overalCond.astype(float)
    yearBuilt = yearBuilt.astype(float)
    yearRemodAdd = yearRemodAdd.astype(float)
    grLivArea = grLivArea.astype(float)
    monthSold = np.array([a[0] if len(a[0]) == 2 else "0"+a[0] for a in monthSold])
    dateSold = np.array([monthSold[i] + yearSold[i][0] for i in range(len(yearSold))]).astype(float)
    dateSold = dateSold[np.newaxis].T

    # create vectors of input + result(output)
    data_vector = np.concatenate((neighborhoodType, bldgType, houseStyle, neighborhoodType, overallQual, overalCond, yearBuilt,
                    yearRemodAdd, grLivArea, dateSold, targetPrice), axis=1)

    return data_vector

