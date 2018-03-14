import pandas as pd
import numpy as np

fuzzy_set_size = 26
fuzzy_distance = 0.02

fuzzy_set = np.zeros(fuzzy_set_size)
for i in range(0, fuzzy_set_size):
    fuzzy_set[i] = fuzzy_distance * i + fuzzy_distance / 2
def fuzzy(training_set):

    length_x_train = training_set.size
    difference = np.zeros(length_x_train - 1)

    for i in range(0, length_x_train - 1):
        difference[i] = training_set[i + 1] - training_set[i] + 0.25

    fuzzy_result = np.zeros([length_x_train - 1, fuzzy_set_size])
    for i in range(1, length_x_train - 2):
        j = int(difference[i] / fuzzy_distance)
        fuzzy_result[i][j] = 1
        fuzzy_result[i][j - 1] = (fuzzy_set[j + 1] + fuzzy_set[j] - 2 * difference[i]) / (2 * fuzzy_distance)
        fuzzy_result[i][j + 1] = (- fuzzy_set[j - 1] - fuzzy_set[j] + 2 * difference[i]) / (2 * fuzzy_distance)
    df = pd.DataFrame(data = fuzzy_result)
    df.to_csv('x_train.csv')

    return fuzzy_result

def defuzzy(testing_set, training_result):
    df = pd.DataFrame(data=training_result)
    df.to_csv('y_predict.csv')
    print 'ok'
    length_y_test = testing_set.size
    y_prid = np.zeros([length_y_test - 1])
    for i in range (0, length_y_test - 1):
        tu = 0
        mau = 0

        for j in range (0, fuzzy_set_size):
            tu = tu + fuzzy_set[j] *  training_result[i][j]
            mau = mau + training_result[i][j]
        difference = tu / mau - 0.25
        y_prid[i] = testing_set[i] + difference

    return y_prid

