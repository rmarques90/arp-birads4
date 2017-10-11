print(__doc__)
#import library

import pandas as pd
import numpy as np
import pylab as pl

from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn import svm, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix as cfm, precision_score as ps, recall_score as rs, f1_score as f1s


#import data

data = pd.read_csv("/home/fox/Documents/Base hospital/BI-RADS_4.csv", header=0)
data.info()

#Create targets

y = np.array(data.Tumor)
labels = LabelEncoder()
target = labels.fit_transform(y)

#create features

cols = data.columns[(data.columns != 'Tumor')]
features = data[cols]
features = (features - features.mean()) / (features.std())

fts = np.array(data[cols])

#split data in train and test

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=42)
print "x_train size:", len(x_train)
print "x_test size:", len(x_test)

#Test to know the best KNN for the database

knn = KNeighborsClassifier()

k_range = list(range(1,20))
k_scores = []

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn, x_train, y_train, cv=10)
    k_scores.append(scores.mean())
print(np.round(k_scores, 3))

plt.plot(k_range, k_scores, color='red')
plt.xlabel('Valores de K')
plt.ylabel('Recall')
plt.show()

#knn test

neighbors = [3, 5, 7, 13, 1]

for n in range(0,5):
    print "Quantidade Vizinhos:", neighbors[n]
    knn3 = KNeighborsClassifier(n_neighbors=neighbors[n])
    knn3.fit(x_train, y_train)

    print "Accuracy Training KNN:", knn3.score(x_train, y_train)

    predictions = knn3.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, predictions)

    print "Accuracy Test KNN:", accuracy
    print "Matriz de Confusao KNN:"
    print cfm(y_test, predictions)
    print "F1 score KNN:"
    print f1s(y_test, predictions)
    print "Precision score KNN:"
    print ps(y_test, predictions)
    print "Recall score KNN:"
    print rs(y_test, predictions)


#svm kernel linear

svm = svm.SVC(kernel='linear', C=1.0)
svm.fit(x_train,y_train)

predictionsSvm = svm.predict(x_test)

accuracySvm = metrics.accuracy_score(predictionsSvm, y_test)

print "SVM LINEAR Accuracy Test:", accuracySvm

print "Matriz de Confusao SVM LINEAR:"
print cfm(y_test, predictionsSvm)
print "F1 score SVM LINEAR:"
print f1s(y_test, predictionsSvm)
print "Precision score SVM LINEAR:"
print ps(y_test, predictionsSvm)
print "Recall score SVM LINEAR:"
print rs(y_test, predictionsSvm)

# get the separating hyperplane
w = svm.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (svm.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = svm.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = svm.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
            s=80, facecolors='None')
plt.scatter(fts[:, 0], fts[:, 1],c=target, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()

#svm kernel poly
#
# svmPoly = svm.SVC(kernel='poly')
# svmPoly.fit(x_train,y_train)
#
# predictionsSvmPoly = svmPoly.predict(x_test)
#
# accuracySvmPoly = metrics.accuracy_score(predictionsSvmPoly, y_test)
#
# print "SVM POLY Accuracy Test:", accuracySvmPoly
#
# print "Matriz de Confusao SVM POLY:"
# print cfm(predictionsSvmPoly, y_test)

#svm kernel rbf

# svm = svm.SVC(kernel='rbf')
# svm.fit(x_train,y_train)
#
# predictionsSvm = svm.predict(x_test)
#
# accuracySvm = metrics.accuracy_score(predictionsSvm, y_test)
#
# print "SVM RBF Accuracy Test:", accuracySvm
#
# print "Matriz de Confusao SVM RBF:"
# print cfm(predictionsSvm, y_test)

#svm kernel sigmoid

# svm = svm.SVC(kernel='sigmoid')
# svm.fit(x_train,y_train)
#
# predictionsSvm = svm.predict(x_test)
#
# accuracySvm = metrics.accuracy_score(predictionsSvm, y_test)
#
# print "SVM SIGMOID Accuracy Test:", accuracySvm
#
# print "Matriz de Confusao SVM SIGMOID:"
# print cfm(predictionsSvm, y_test)

#naive Bayes

nB = GaussianNB()

fit = nB.fit(x_train, y_train)

print "NB Accuracy Train:", fit

predictionsNB = nB.predict(x_test)

accuracyNB = metrics.accuracy_score(predictionsNB, y_test)

print "NB Accuracy Test:", accuracyNB

print "Matriz de Confusao NB:"
print cfm(y_test, predictionsNB)
print "F1 score NB:"
print f1s(y_test, predictionsNB)
print "Precision score NB:"
print ps(y_test, predictionsNB)
print "Recall score NB:"
print rs(y_test, predictionsNB)