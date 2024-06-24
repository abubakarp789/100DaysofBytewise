import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

import matplotlib.pyplot as plt

#(a). Load the Iris dataset
iris = load_iris()
X_data = iris.data
y_data = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

#(b). Train a decision tree classifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_train, y_train)

#(c). Print the classification report and confusion matrix
y_predict = decision_tree_classifier.predict(X_test)

print("Classification Report of Iris Dataset:")
print(classification_report(y_test, y_predict))

print("Confusion Matrix of Iris Dataset:")
print(confusion_matrix(y_test, y_predict))

#(d). Visualize the decision tree
fig = plt.figure(figsize=(10, 8))
_ = tree.plot_tree(decision_tree_classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree of Iris Dataset")
plt.show()