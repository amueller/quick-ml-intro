import pandas as pd
from sklearn.preprocessing import StandardScaler
# we're setting some options for nicer printing here
np.set_printoptions(suppress=True, precision=4)

data = pd.read_csv("boston_house_prices.csv")
X = data.drop("MEDV", axis=1)
y = data.MEDV

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train_scaled mean:")
print(X_train_scaled.mean(axis=0))
print("X_test_scaled mean:")
print(X_test_scaled.mean(axis=0))
