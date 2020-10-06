# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:02:50 2020

@author: Nurullah
"""
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#2.1. Veri Yukleme
veriler = pd.read_csv('term-deposit-marketing-2020.csv')

#veri on isleme
X= veriler.iloc[:,0:13].values
Y = veriler.iloc[:,13].values

#encoder:  Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])
le3 = LabelEncoder()
X[:,3] = le3.fit_transform(X[:,3])
le4 = LabelEncoder()
X[:,4] = le4.fit_transform(X[:,4])

le6 = LabelEncoder()
X[:,6] = le6.fit_transform(X[:,6])
le7 = LabelEncoder()
X[:,7] = le7.fit_transform(X[:,7])
le8 = LabelEncoder()
X[:,8] = le8.fit_transform(X[:,8])
le10 = LabelEncoder()
X[:,10] = le10.fit_transform(X[:,10])

le11 = LabelEncoder()
Y = le11.fit_transform(Y)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25, random_state=1)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#3 Yapay Sinir ağı
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(13, kernel_initializer = 'uniform', activation = 'relu' , input_dim = 13))
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['accuracy'] )
hist=classifier.fit(X_train, y_train, epochs=40)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

cm = confusion_matrix(y_test,y_pred)
class_names = ['TRUE', 'FALSE']

fig, ax = plot_confusion_matrix(cm,class_names=class_names)
plt.show()

#ROC Curve and AUC Score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
x_test = np.asarray(x_test).astype(np.float32)
probs = classifier.predict_proba(x_test)
probs = y_pred
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)
print("AVG Accuracy")
print(np.mean(hist.history['accuracy']))