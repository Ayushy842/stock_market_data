import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

stock_symbol = "IOC.NS"  # Replace with the desired stock symbol
start_date = datetime.today() - timedelta(days=365)  # Start date: 1 year ago from today
end_date = datetime.today()  # End date: today

# Fetch stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Display the stock data
print(stock_data)
# Extract the "Close" prices as the target variable
data = stock_data[['Close']].copy()
# Create additional features, such as moving averages or technical indicators, if desired

# Shift the target variable by 30 days to create the prediction target
data['Prediction'] = data['Close'].shift(-30)

# Drop any rows with missing values
data.dropna(inplace=True)

# Separate the features and target variable
X = data.drop('Prediction', axis=1)
y = data['Prediction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
