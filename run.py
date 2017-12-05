import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from model import DataManager, Classifier

REPETITION = 30

probes, result = DataManager.load_data('data/diabetes.arff')
x = np.array(probes)
y = np.array(result)

multinomial = Classifier(MultinomialNB(), REPETITION, x, y)
multinomial.calculate_indicators()

logistic_regression = Classifier(LogisticRegression(), REPETITION, x, y)
logistic_regression.calculate_indicators()

kneighbours_classifier = Classifier(KNeighborsClassifier(5), REPETITION, x, y)
kneighbours_classifier.calculate_indicators()

mlp_classifier = Classifier(MLPClassifier(), REPETITION, x, y)
mlp_classifier.calculate_indicators()

