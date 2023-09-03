import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from word2number import w2n
import math
import joblib

df = pd.read_csv('data.csv')
print(df)

reg = linear_model.LinearRegression()

x = df[['T_Supply', 'T_Return', 'SP_Return', 'T_Saturation', 'T_Outdoor', 'RH_Supply', 'RH_Return', 'RH_Outdoor'
        ]]
y = df[['Energy', 'Power']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
reg.fit(x_train, y_train)
print(reg.predict(x_test))

print(reg.score(x_test, y_test))
