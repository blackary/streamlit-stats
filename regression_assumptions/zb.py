import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

"# Regression Assumptions"

df = pd.read_csv("regression_output/housing.csv")

df

Y_col = "SalePrice"
X_col = ["LotArea"]

y = df[Y_col]
X = df[X_col]

st.write(df[X_col + [Y_col]].describe())
# st.write(y.describe())

"## Linearity"

reg = LinearRegression()

reg.fit(X, y)

preds = reg.predict(X)

fig, ax = plt.subplots()

ax.set_title("Actuals vs predictions")
ax.scatter(preds, y)

st.write(fig)

fig, ax = plt.subplots()

residuals = preds - y

ax.set_title("Residuals vs predictions")
ax.scatter(preds, residuals)

st.write(fig)

y = np.log(y)
X = np.log(X)

reg = LinearRegression()

reg.fit(X, y)

preds = reg.predict(X)

fig, ax = plt.subplots()

ax.set_title("Actuals vs predictions")
ax.scatter(preds, y)

st.write(fig)

fig, ax = plt.subplots()

residuals = preds - y
ax.set_title("Residuals vs predictions")
ax.scatter(preds, residuals)

st.write(fig)
