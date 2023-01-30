
# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] 
# Load the data
df = pd.read_csv('iris.data', names=columns)
df.head()
df.shape
df.info()
df.describe()
df['Class_labels'].value_counts()
sns.violinplot(y='Class_labels', x='Sepal length', data=df, inner='quartile')
plt.show()
sns.violinplot(y='Class_labels', x='Sepal width', data=df, inner='quartile')
plt.show()
sns.pairplot(df, hue='Class_labels', markers='*')
plt.show()
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True, cmap='cubehelix_r')
plt.show()
X = df.drop(['Class_labels'], axis=1)
y = df['Class_labels']
print(f'X shape: {X.shape} | y shape: {y.shape} ')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC(gamma='auto')))
# evaluate each model in turn
results = []
model_names = []
for name, model in models:
            kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
            results.append(cv_results)
            model_names.append(name)
            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
model = SVC(gamma='auto')
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test, prediction)}')
print(f'Classification Report: \n {classification_report(y_test, prediction)}')
