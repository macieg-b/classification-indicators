from datetime import datetime
import random

import numpy as np
from scipy.io import arff
from sklearn.metrics import confusion_matrix, roc_auc_score


class Classifier:
    def __init__(self, classifier, n, x, y):
        self.__classifier = classifier
        self.__n = n
        self.__x = x
        self.__y = y

        self.__fit_time_array = []
        self.__predict_time_array = []

        self.__acc_array = []
        self.__sens_array = []
        self.__spec_array = []
        self.__prec_array = []
        self.__f1_array = []
        self.__balanced_acc_array = []
        self.__auc_array = []

    def calculate_indicators(self):
        teach_data, test_data, teach_result, test_result = DataManager.split_data(self.__x, self.__y, 0.5)
        for i in range(0, self.__n):
            start_time = datetime.now()
            self.__classifier.fit(teach_data, teach_result)
            tick_time = datetime.now()
            prediction = self.__classifier.predict(test_data)
            tock_time = datetime.now()
            self.__fit_time_array.append(tick_time - start_time)
            self.__predict_time_array.append(tock_time - tick_time)
            decision = self.__classifier.predict_proba(test_data)

            pn, np, nn, pp = confusion_matrix(test_result, prediction).ravel()
            accuracy = (pp + pn) / (pp + pn + np + nn)
            sensitivity = pp / (pp + nn)
            specificity = pn / (np + pn)
            precision = pp / (pp + np)
            self.__acc_array.append(accuracy)
            self.__sens_array.append(sensitivity)
            self.__spec_array.append(specificity)
            self.__prec_array.append(precision)
            self.__f1_array.append(2 * (precision * sensitivity) / (precision + sensitivity))
            self.__balanced_acc_array.append(0.5 * (sensitivity + specificity))
            self.__acc_array.append(roc_auc_score(test_result, decision[:, 1]))

    def get_mean_result(self):
        return np.mean(self.__acc_array), np.mean(self.__sens_array), np.mean(self.__spec_array), np.mean(self.__prec_array), \
               np.mean(self.__f1_array), np.mean(self.__balanced_acc_array), np.mean(self.__auc_array), \
               np.mean(self.__fit_time_array), np.mean(self.__predict_time_array)


class DataManager:
    def __init__(self):
        pass

    @staticmethod
    def load_data(file_path):
        data, meta = arff.loadarff(file_path)
        x_data = []
        y_data = []
        for w in range(len(data)):
            x_data.append([])
            for k in range(len(data[0])):
                if k == (len(data[0]) - 1):
                    y_data.append(data[w][k])
                else:
                    x_data[w].append(data[w][k])
        return x_data, y_data

    @staticmethod
    def split_data(x, y, ratio):
        random.shuffle(x)
        teach_size = int(len(x) * ratio)
        return x[:teach_size], x[teach_size:], y[:teach_size], y[teach_size:]