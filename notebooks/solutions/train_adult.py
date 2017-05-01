import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("adult.csv", index_col=0)
# optionally drop columns
# columns = ['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']
# data = data[columns]

X = data.drop("income", axis=1)
y = data.income.values

X_dummies = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Training score")
print(logreg.score(X_train, y_train))
print("Test score")
print(logreg.score(X_test, y_test))
