import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = datasets.load_iris()

data = iris.data
target = iris.target

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

# Programaticaly find best K
scores = []
for k in range(1, 50):
  model = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute')
  score = cross_val_score(model, data_train, target_train, scoring ='accuracy')
  scores.append(score.mean())

# Find min error from scores
errors = [1 - x for x in scores]
optimal_k = errors.index(min(errors))

print('Optimum K for this dataset: ' + str(optimal_k))

# Calc new model using optimal N
model = KNeighborsClassifier(n_neighbors = optimal_k)
model.fit(data_train, target_train)
predictions = model.predict(data_test)

accuracy = accuracy_score(target_test, predictions) * 100

print('Model Accuracy: %.2f%%' % accuracy)
print('Classification Report:')
print(classification_report(target_test, predictions))
###### End

###### Plot the orignal dataset and the result of the KNN model
sepal_measurements = data_test[:, :2]
petal_measurements = data_test[:, 2:4]

plt.subplot(2, 2, 1)
plt.title('Unclassified Data Grouped By Sepal Characteristics')
plt.scatter(sepal_measurements[:, 0], sepal_measurements[:, 1], c=target_test, cmap='gist_rainbow')
plt.xlabel('Sepal Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)
plt.plot()

plt.subplot(2, 2, 2)
plt.title('Unclassified Data Grouped By Petal Characteristics')
plt.scatter(petal_measurements[:, 0], petal_measurements[:, 1], c=target_test, cmap='gist_rainbow')
plt.xlabel('Petal Length', fontsize=18)
plt.ylabel('Petal Width', fontsize=18)
plt.plot()

plt.subplot(2, 2, 3)
plt.title('Classified Data Grouped By Sepal Characteristics')
plt.scatter(sepal_measurements[:,0], sepal_measurements[:,1], c=predictions, cmap='gist_rainbow')
plt.xlabel('Sepal Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)
plt.plot()

plt.subplot(2, 2, 4)
plt.title('Classified Data Grouped By Petal Characteristics')
plt.scatter(petal_measurements[:,0], petal_measurements[:,1], c=predictions, cmap='gist_rainbow')
plt.xlabel('Petal Length', fontsize=18)
plt.ylabel('Petal Width', fontsize=18)
plt.plot()

plt.tight_layout()
plt.show()
