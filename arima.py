import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
stock_symbol = "IOC.NS"  # Replace with the desired stock symbol
start_date = "2022-06-30"  # Replace with the desired start date
end_date = "2023-07-30"  # Replace with the desired end date

# Fetch stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Extract the "Close" prices
data = stock_data[['Close']].copy()

# Display the stock data
print(data)
# Convert the "Close" prices to a univariate time series
ts = data['Close']

# Split the data into training and testing sets
train_data = ts[:int(0.8 * len(ts))]
test_data = ts[int(0.8 * len(ts)):]

# Plot the training and testing data
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Testing Data')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Data')
plt.legend()
plt.show()
# Create and fit the ARIMA model
model = ARIMA(train_data, order=(2, 1, 2))
model_fit = model.fit()

# Get the predictions
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Convert the predictions to a Pandas Series with proper index
predictions.index = test_data.index

# Plot the predictions and the actual values
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Testing Data')
plt.plot(predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('ARIMA Predictions')
plt.legend()
plt.show()
# Calculate the root mean squared error (RMSE)
rmse = ((predictions - test_data) ** 2).mean() ** 0.5
print("Root Mean Squared Error (RMSE):", rmse)
