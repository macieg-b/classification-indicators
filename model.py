from datetime import datetime
import random

import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import matplotlib.pyplot as plt


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

        self.__teach_data = []
        self.__test_data = []
        self.__teach_result = []
        self.__test_result = []
        self.__decision = []

    def calculate_indicators(self):
        for i in range(self.__n):
            self.__teach_data, self.__test_data, self.__teach_result, self.__test_result = \
                train_test_split(self.__x, self.__y, test_size=0.5, random_state=i)

            start_time = datetime.now()
            self.__classifier.fit(self.__teach_data, self.__teach_result)
            tick_time = datetime.now()
            prediction = self.__classifier.predict(self.__test_data)
            tock_time = datetime.now()
            self.__fit_time_array.append(tick_time - start_time)
            self.__predict_time_array.append(tock_time - tick_time)
            self.__decision = self.__classifier.predict_proba(self.__test_data)

            pn, np, nn, pp = confusion_matrix(self.__test_result, prediction).astype(float).ravel()
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
            self.__auc_array.append(roc_auc_score(self.__test_result, self.__decision[:, 1]))

    def get_mean_result(self):
        return np.mean(self.__acc_array), \
               np.mean(self.__sens_array), \
               np.mean(self.__spec_array), \
               np.mean(self.__prec_array), \
               np.mean(self.__f1_array), \
               np.mean(self.__balanced_acc_array), \
               np.mean(self.__auc_array), \
               np.mean(self.__fit_time_array), \
               np.mean(self.__predict_time_array)

    def get_chart_data(self):
        fpr, tpr, _ = roc_curve(self.__test_result, self.__decision[:, 1])
        return fpr, tpr


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

    @staticmethod
    def get_nearest(points, cords):
        dists = [(pow(point[0] - cords[0], 2) + pow(point[1] - cords[1], 2), point)
                 for point in points]
        nearest = min(dists)
        return nearest[1]


class PlotGenerator:
    @staticmethod
    def roc_chart(classifiers, names):
        plt.figure(1)

        i = 0
        for arg in classifiers:
            fpr, tpr = arg.get_chart_data()
            plt.plot(fpr, tpr, label=names[i])
            nearest = DataManager.get_nearest(zip(fpr, tpr), (0, 1))
            plt.plot(nearest[0], nearest[1], 'r*')
            i += 1
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc='best')
        plt.show()
