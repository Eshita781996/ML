import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create a simple dataset
data = {
    'Square_Feet': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'Bedrooms': [3, 3, 3, 3, 2, 3, 4, 4, 3, 3],
    'Price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}

df = pd.DataFrame(data)

# Features (independent variables) and target variable (Price)
X = df[['Square_Feet', 'Bedrooms']]
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Visualize the results (optional)
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.title('Actual vs Predicted Prices')
# plt.show()
#1400 ,3,245000

test_data = {
    'Square_Feet': [1400],
    'Bedrooms': [3]
}
df_test = pd.DataFrame(test_data)
print(model.predict(df_test))

with open('linearregression.pickle', 'wb') as file:
    pickle.dump(model, file)
