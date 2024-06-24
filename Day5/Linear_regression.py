import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

boston_dataset = pd.read_csv('Day5/BostonHousing.csv')
#print(boston_dataset.keys())
boston_dataset.keys()
#print(boston_dataset.head())
boston_dataset = pd.DataFrame(boston_dataset._data, columns=boston_dataset.columns)
#print(boston_dataset.isnull().sum())
sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
sns.histplot(boston_dataset['medv'], bins=30, kde=True)
plt.show()
correlation_matrix = boston_dataset.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

plt.figure(figsize=(20, 5))

features = ['lstat', 'rm']
target = boston_dataset['medv']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston_dataset[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('medv')

plt.show()

X = pd.DataFrame(np.c_[boston_dataset['lstat'], boston_dataset['rm']], columns = ['lstat','rm'])
Y = boston_dataset['medv']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

y_train = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train)))
r2 = lin_model.score(X_train, Y_train)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test = lin_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test)))
# r-squared score of the model
r2 = lin_model.score(X_test, Y_test)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# Predict
y_pred = lin_model.predict(X_test)
print(y_pred.shape)
print(y_pred)

sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Actual Data points')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Line')
plt.legend()
plt.show()





