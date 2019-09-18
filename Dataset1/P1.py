from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import DistanceMetric
#from pandas.tools.plotting import scatter_matrix
import importlib
import pandas as pd
from matplotlib.animation import adjusted_figsize
from pandas.plotting._matplotlib.misc import scatter_matrix
pd.set_option('display.width',None)
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()

#Data description
pima = pd.read_csv("E:\Pythontest\diabetes.csv") 
print(pima.head())
pima.info(verbose=True)
print(pima.describe().T)
print(" ")

#replace Non data
pima_copy = pima.copy(deep = True)
pima_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = pima_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.NaN)
print(pima_copy.isnull().sum())

pima_copy['Glucose'].fillna(pima_copy['Glucose'].mean(), inplace = True)
pima_copy['BloodPressure'].fillna(pima_copy['BloodPressure'].mean(), inplace = True)
pima_copy['SkinThickness'].fillna(pima_copy['SkinThickness'].median(), inplace = True)
pima_copy['Insulin'].fillna(pima_copy['Insulin'].median(), inplace = True)
pima_copy['BMI'].fillna(pima_copy['BMI'].median(), inplace = True)


#data visualization
pima.hist(figsize = (20,20))

pima_copy.hist(figsize = (20,20))

scatter_matrix(pima_copy,figsize=(25,25))

sns.pairplot(pima_copy,hue='Outcome')

plt.figure(figsize=(10,10))
p=sns.heatmap(pima.corr(), annot = True, cmap = 'RdYlGn') 
plt.show()



#standardization
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(pima_copy.drop(["Outcome"], axis = 1),), columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = pima_copy.Outcome
print(X.head())

#machine learning 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)

test_scores = []
train_scores = []

#distance
knn2 = KNeighborsClassifier(p = 1, metric='minkowski')
knn3 = KNeighborsClassifier(p = 2, metric='minkowski')
for i in range(1,20):

    knn1 = KNeighborsClassifier(i)
    knn1 = KNeighborsClassifier(i,p = 2, metric='minkowski')
    knn1.fit(X_train,y_train)
    
    train_scores.append(knn1.score(X_train,y_train))
    test_scores.append(knn1.score(X_test,y_test))


#training   
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
#testing
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,20),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,20),test_scores,marker='o',label='Test Score')
plt.show()

y_pred = knn1.predict(X_test)
confusion_matrix(y_test,y_pred)
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

print(classification_report(y_test,y_pred))
