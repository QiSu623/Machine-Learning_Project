# -*- coding: utf-8 -*-
'''
Created on 2019��9��15��

@author: Qi_Su
'''
import numpy as np 
import pandas as pd
pd.set_option('display.width',None)
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
from sklearn.model_selection import train_test_split

data_dir = "E:/Pythontest/"

# load csv files to numpy arrays
def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + "train.csv")
    print(train.shape)
    X_train = train.values[0:train_row,1:] 
    y_train = train.values[0:train_row,0] 
    
    
    Pred_test = pd.read_csv(data_dir + "test.csv").values  
#     print(Pred_test.shape)
#     print(pd.read_csv(data_dir + "test.csv").head())
    return X_train, y_train, Pred_test

train_row = 42000
Origin_X_train, Origin_y_train, Origin_X_test = load_data(data_dir, train_row)

print(" ")
print(Origin_X_train.shape, Origin_y_train.shape, Origin_X_test.shape)
print(Origin_X_train)

row = 4
# print (X_train[row].reshape((28, 28)))

print (Origin_y_train[row])

plt.imshow(Origin_X_train[row].reshape((28, 28)))
plt.show()


classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rows = 10

print(classes)
for y, cls in enumerate(classes):
    idxs = np.nonzero([i == y for i in Origin_y_train])
    idxs = np.random.choice(idxs[0], rows)
    for i , idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1
        plt.subplot(rows, len(classes), plt_idx)
        plt.imshow(Origin_X_train[idx].reshape((28, 28)))
        plt.axis("off")
        if i == 0:
            plt.title(cls)
        

plt.show()

#testing and training
X_train,X_vali, y_train, y_vali = train_test_split(Origin_X_train,
                                                   Origin_y_train,
                                                   test_size = 0.25,
                                                   random_state = 0)


print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)

#knn
k_range = range(1, 20)
scores = []


for k in k_range:
    print("k = " + str(k) + " begin ")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_vali)
    accuracy = accuracy_score(y_vali,y_pred)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_vali, y_pred))  
    print(confusion_matrix(y_vali, y_pred))  
    
    print("Complete time: " + str(end-start) + " Secs.")

print (scores)
plt.plot(k_range,scores)
plt.xlabel('Value of K')
plt.ylabel('Testing accuracy')
plt.show()


k = 5

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(Origin_X_train,Origin_y_train)
y_pred = knn.predict(Origin_X_test[:5000])


print (y_pred[500])
plt.imshow(Origin_X_test[500].reshape((28, 28)))
plt.show()