# Imports
#Don't forget to run: pip install -r requirments.txt
import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

df = quandl.get("WIKI/AMZN")  # Amazon Stocks

#Adding more features for better results
df['MA10'] = df['Adj. Close'].rolling(window=10).mean()
df['MA50'] = df['Adj. Close'].rolling(window=50).mean()
df['Daily Return'] = df['Adj. Close'].pct_change()


forecast_out = 30
df["Prediction"] = df[["Adj. Close"]].shift(
    -forecast_out
)  # Creates a column for predicitions
df = df.dropna()

X = np.array(
    df.drop(["Prediction"], axis=1)
)  # array of all the elements except predition "y"
X = X[:-forecast_out]

Y = np.array(df["Prediction"])
Y = Y[:-forecast_out]

# Handling missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Standardizing the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Splitting Dataset, %20 test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# Create and Train 1: SUPPORT VECTOR MACHINE (Regressor)
# Hyperparameter tuning for SVR
param_grid = {
    'C': [1e2, 1e3, 1e4],
    'gamma': [0.01, 0.1, 1]
}

grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5)
grid_search.fit(x_train, y_train)

print("Best parameters for SVR: ", grid_search.best_params_)
svr_rbf = grid_search.best_estimator_

# Testing the SVR model's accuracy
svr_scores = cross_val_score(svr_rbf, x_train, y_train, cv=5)
svm_confidence = svr_rbf.score(x_test, y_test)
print("SVR Cross-Validation Scores: ", svr_scores)
print("SVM Confidence: ", svm_confidence)
print("\n")

# Create and Train 2: Lineer Regression Model
LR = LinearRegression()

LR.fit(x_train, y_train)

# Testing Model 2 Accuracy, coefficient of the determination R², best: 1.0
lr_scores = cross_val_score(LR, x_train, y_train, cv=5)
LR_confidence = LR.score(x_test, y_test)
print("LR Cross-Validation Scores: ", lr_scores)
print("LR Confidence: ", LR_confidence)
print("\n")

# Predicition
x_forecast = scaler.transform(imputer.transform(np.array(df.drop(['Prediction'], axis=1))[-forecast_out:]))  # Apply imputer and scaler to forecast data


# Print SVM predictions for the next 30 days
svm_predition = svr_rbf.predict(x_forecast)
print("SVM_Predicition")
print(svm_predition)
print("\n")

# Print LR predictions for the next 30 days
LR_Prediction = LR.predict(x_forecast)
print("LR_Prediction")
print(LR_Prediction)

last_date = df.index[-1]
prediction_dates = pd.date_range(last_date, periods=forecast_out+1).tolist()[1:]

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(prediction_dates, svm_predition, label='SVM Predictions', color='blue', marker='o')
plt.plot(prediction_dates, LR_Prediction, label='Linear Regression Predictions', color='red', marker='x')
plt.title('Stock Price Predictions for the Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Predicted Stock Price')
plt.legend()
plt.grid(True)
plt.xticks(prediction_dates, rotation=45)
plt.tight_layout()

# Ensure that the x-axis has 30 ticks
plt.locator_params(axis='x', nbins=30)

plt.show()