import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("boston_house_prices.csv")
print("Number of samples: %d  number of features: %d"
      % (data.shape[0], data.shape[1]))
print("Columns:")
print(data.columns)

X = data.drop("MEDV", axis=1)
y = data.MEDV

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape)

# plotting average room number RM vs MEDV
data.plot("RM", "MEDV", kind="scatter")
