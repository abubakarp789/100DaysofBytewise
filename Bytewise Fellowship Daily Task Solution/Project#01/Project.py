# Step 1: Data Collection and Preparation
import pandas as pd

# Load the dataset
file_path = 'Bytewise Fellowship Daily Task Solution\Project#01\Historical_Data.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Step 2: Data Cleaning
# Inspect the dataset
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Drop duplicates
data = data.drop_duplicates()

# Feature Engineering: Extract year and month from the 'Date' column
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Updated data preview
print(data.head())

# Step 3: Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Sales distribution
sns.histplot(data['Sold_Units'], kde=True)
plt.title('Sales Distribution')
plt.show()

# Sales over time
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Sold_Units'])
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sold Units')
plt.show()

# Correlation Analysis
numeric_data = data.select_dtypes(include='number')
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# Step 4: Probability Distributions
# Probability Distribution
sns.kdeplot(data['Sold_Units'])
plt.title('Sold Units Probability Distribution')
plt.show()

# Q-Q Plot to check normality
from scipy import stats
import numpy as np

stats.probplot(data['Sold_Units'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Sold Units')
plt.show()

# Step 5: Linear Regression Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

print('Mean Squared Error:', mse)
print('R-squared:', r2)

# Step 6: Interpretation and Reporting
# Interpret the results
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Visualize the predictions
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sold Units')
plt.ylabel('Predicted Sold Units')
plt.title('Actual vs Predicted Sold Units')
plt.show()
