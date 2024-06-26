import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

np.random.seed(42)

dataframe = pd.read_csv("/Users/andrey_matrusov/Desktop/TRAIN.csv", sep=",", index_col=0)

'''enc = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

for heading in ["cut", "color", "clarity"]:
    transformed = enc.fit_transform(dataframe[[heading]])
    dataframe = pd.concat([transformed, dataframe], axis=1).drop(columns=heading)'''

enc = pd.get_dummies(dataframe, columns=["cut", "color", "clarity"])

dataframe = shuffle(enc, random_state=42)

print(dataframe)

X = dataframe.drop(columns="price")
y = dataframe["price"]

#X, y = shuffle(X, y, random_state=42)

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

clf_tree = DecisionTreeRegressor(criterion="squared_error", max_depth=12, random_state=42)

#preds = clf_tree.predict(x_test)
cv = cross_val_score(clf_tree, X, y, cv=10, scoring="r2")
predictions = (sum(cv)/10)

print(predictions)

'''print(dataframe)
print(y)'''
