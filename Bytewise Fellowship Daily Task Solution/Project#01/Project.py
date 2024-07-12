from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
file_path = 'Bytewise Fellowship Daily Task Solution\Project#01\Historical_Data.csv'
data = pd.read_csv(file_path)

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Split the data
X = data[['Year', 'Month']]  # Example features
y = data['Sold_Units']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Interpret the results
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

# Visualize the predictions
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sold Units')
plt.ylabel('Predicted Sold Units')
plt.title('Actual vs Predicted Sold Units')
plt.show()

print('Mean Squared Error:', mse)
print('R-squared:', r2)
print('Coefficients:', coefficients)
