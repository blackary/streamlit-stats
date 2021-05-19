import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("regression_output/housing.csv")

POTENTIAL_X_VARS = ["LotArea", "GrLivArea", "YearBuilt", "SqrtPrice"]
df["SqrtPrice"] = np.sqrt(df["SalePrice"]) + (
    np.random.normal(scale=50, size=df["SalePrice"].shape)
)

X_VARS = {}
st.sidebar.write("# X Variables")
for v in POTENTIAL_X_VARS:
    if st.sidebar.checkbox(v, value=True):
        transformation = st.sidebar.radio(
            "Transformation", ["None", "sqrt", "^2", "log"], key=v + "transformation"
        )
        X_VARS[v] = transformation

st.sidebar.write("---")

st.sidebar.write("# Y Variable")
st.sidebar.write("SalePrice")

y_transformation = st.sidebar.radio(
    "Transformation", ["None", "sqrt", "^2", "log"], key="SalePrice transformation"
)
y_var = ["SalePrice"]


X = df[X_VARS.keys()]
for var, transformation in X_VARS.items():
    if transformation == "None":
        pass
    elif transformation == "sqrt":
        X[var] = np.sqrt(X[var])
    elif transformation == "^2":
        X[var] = X[var] ** 2
    elif transformation == "log":
        X[var] = np.log(X[var])

y = df[y_var]
for var, transformation in [(y_var, y_transformation)]:
    if transformation == "None":
        pass
    elif transformation == "sqrt":
        y[var] = np.sqrt(y[var])
    elif transformation == "^2":
        y[var] = y[var] ** 2
    elif transformation == "log":
        y[var] = np.log(y[var])

for var in X_VARS:
    # st.write()
    fig, ax = plt.subplots()
    ax.scatter(X[var], y[y_var])
    ax.set_title(f"{y_var[0]} vs. {var}")
    st.write(fig)

model = LinearRegression()

model.fit(X, y)
st.write("R^2", model.score(X, y))

y_pred = model.predict(X)

fig, ax = plt.subplots()
ax.scatter(y, y_pred)
ax.scatter(y, y)
ax.set_title("Linear Regression Predicted vs Actual Price")

st.write(fig)
