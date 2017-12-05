import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from model import DataManager, Classifier, PlotGenerator

REPETITION = 50

probes, result = DataManager.load_data('data/diabetes.arff')
x = np.array(probes)
y = np.array(result)

bayes = Classifier(MultinomialNB(), REPETITION, x, y)
bayes.calculate_indicators()

logistic_regression = Classifier(LogisticRegression(), REPETITION, x, y)
logistic_regression.calculate_indicators()

kneighbours_classifier = Classifier(KNeighborsClassifier(10), REPETITION, x, y)
kneighbours_classifier.calculate_indicators()

mlp_classifier = Classifier(MLPClassifier(), REPETITION, x, y)
mlp_classifier.calculate_indicators()

classifiers_array = []
classifiers_array.append(bayes)
classifiers_array.append(logistic_regression)
classifiers_array.append(kneighbours_classifier)
classifiers_array.append(mlp_classifier)

PlotGenerator.roc_chart(classifiers_array, ['Bayes', 'Logistic regression', 'k-Neighbours', 'Neural network'])

