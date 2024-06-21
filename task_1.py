import pandas as pd

# Load the dataset
data = pd.read_csv('train.csv')

# Display the first few rows of the dataset
data.head()
import matplotlib.pyplot as plt
import seaborn as sns

# Pairplot to visualize relationships
sns.pairplot(data)
plt.show()

# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define features and target variable
X = data[['sqft', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
# Example: Predict the price of a house with 2000 sqft, 3 bedrooms, and 2 bathrooms
new_house = pd.DataFrame({'sqft': [2000], 'bedrooms': [3], 'bathrooms': [2]})
predicted_price = model.predict(new_house)

print(f'Predicted Price: {predicted_price[0]}')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('house_prices.csv')

# Display the first few rows of the dataset
print(data.head())

# Pairplot to visualize relationships
sns.pairplot(data)
plt.show()

# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Define features and target variable
X = data[['sqft', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Example: Predict the price of a house with 2000 sqft, 3 bedrooms, and 2 bathrooms
new_house = pd.DataFrame({'sqft': [2000], 'bedrooms': [3], 'bathrooms': [2]})
predicted_price = model.predict(new_house)

print(f'Predicted Price: {predicted_price[0]}')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('house_prices.csv')

# Display the first few rows of the dataset
print(data.head())

# Pairplot to visualize relationships
sns.pairplot(data)
plt.show()

# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Define features and target variable
X = data[['sqft', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Example: Predict the price of a house with 2000 sqft, 3 bedrooms, and 2 bathrooms
new_house = pd.DataFrame({'sqft': [2000], 'bedrooms': [3], 'bathrooms': [2]})
predicted_price = model.predict(new_house)

print(f'Predicted Price: {predicted_price[0]}')