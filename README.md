# Ex.No: 6               HOLT WINTERS METHOD
### Date: 09.06.2026
### NAME : SIVABALAN M
### AIM:
To run the given algorithm and get the respective graph for the given dataset
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('/content/Gold Price (2013-2023).csv')

# Convert price column to numeric
df['Price'] = df['Price'].str.replace(',', '').astype(float)

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Initial exploration
print(df.head())
print(df.describe())

# Set date as index
df.set_index('Date', inplace=True)

# Resample to monthly frequency (Beginning of month)
monthly_data = df['Price'].resample('MS').mean()

# Plot time series data
plt.figure(figsize=(10,5))
plt.plot(monthly_data)
plt.title("Monthly Gold Price Time Series")
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.show()

# -----------------------------
# Time Series Decomposition
# -----------------------------

decomposition = seasonal_decompose(monthly_data, model='additive', period=12)

decomposition.plot()
plt.show()

# -----------------------------
# Holt-Winters Model
# -----------------------------

model = ExponentialSmoothing(
    monthly_data,
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

# Predictions
predictions = model.fittedvalues

# Future forecast (next 12 months)
future_forecast = model.forecast(12)

# -----------------------------
# RMSE Calculation
# -----------------------------

rmse = np.sqrt(mean_squared_error(monthly_data, predictions))

print("RMSE:", rmse)

# -----------------------------
# Mean and Standard Deviation
# -----------------------------

mean_value = np.mean(monthly_data)
std_value = np.std(monthly_data)

print("Mean:", mean_value)
print("Standard Deviation:", std_value)

# -----------------------------
# Plot Original vs Prediction
# -----------------------------

plt.figure(figsize=(10,5))
plt.plot(monthly_data, label='Original Data')
plt.plot(predictions, label='Fitted Values')
plt.plot(future_forecast, label='Future Forecast', linestyle='--')
plt.title("Gold Price Forecast using Holt-Winters")
plt.legend()
plt.show()


```
### OUTPUT:


### TEST_PREDICTION


<img width="1175" height="846" alt="image" src="https://github.com/user-attachments/assets/fe8c248d-3bf7-4d45-88d0-537b4db6fd69" />

<img width="828" height="556" alt="image" src="https://github.com/user-attachments/assets/5f33917b-a097-4f22-b537-cc58a13c5a18" />

### FINAL_PREDICTION


<img width="985" height="575" alt="image" src="https://github.com/user-attachments/assets/ca151985-a9ca-442a-87f9-acc6e65cfca6" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
