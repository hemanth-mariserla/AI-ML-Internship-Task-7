import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
import matplotlib
matplotlib.use('TkAgg')

cancer = datasets.load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train)

svm_rbf = SVC(kernel='rbf', C=1, gamma=0.5)
svm_rbf.fit(X_train, y_train)

param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 0.5, 1]}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train, y_train)

scores = cross_val_score(SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma']),
                         X_train, y_train, cv=5)

print("Linear Kernel Accuracy:", svm_linear.score(X_test, y_test))
print("RBF Kernel Accuracy:", svm_rbf.score(X_test, y_test))
print("Best Params (RBF):", grid.best_params_)
print("Cross-validation Accuracy:", scores.mean())

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_decision_regions(X_train, y_train, clf=svm_linear, legend=2)
plt.title("SVM with Linear Kernel")

plt.subplot(1,2,2)
plot_decision_regions(X_train, y_train, clf=svm_rbf, legend=2)
plt.title("SVM with RBF Kernel")
plt.show()
