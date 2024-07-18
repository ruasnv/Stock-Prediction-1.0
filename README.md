# Stock Prediction Project

## Overview

I have been amazed by how finance interacts with AI technologies. My journey started off by questioning, "Now that we have powerful prediction models, is it possible to predict stocks?" and followed with, "How would this affect the economy?" Based on these principles, I decided to try a little project to see how it would go. I used Support Vector Regression (SVR) and Linear Regression. The project utilizes historical stock data from Quandl for Amazon and provides predictions for the next 30 days.

## Features

- Fetches historical stock data from Quandl.
- Creates additional features such as moving averages and daily returns.
- Implements Support Vector Regression and Linear Regression models.
- Uses GridSearchCV for hyperparameter tuning.
- Provides visualizations of the predictions.

## Notes

This project is great for beginners like myself who want to experience more with models and libraries. The base code is ideal for trying out new approaches (data tuning, data manipulation, etc.), learning about Quandl, NumPy, pandas, and Matplotlib.

## Needed Improvements

- Stock data isn't up to date on Quandl; we need a new source of data.
- Needs comparison with real-world data (for the prices with older dates).
- Increasing the prediction window to more than 30 days might result in higher accuracy.
- Visualization is weak; it needs a web interface.
- A CI/CD model would be great.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

