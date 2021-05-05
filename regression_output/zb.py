import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
from numpy.random import MT19937, RandomState, SeedSequence
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LinearRegression

# Try 19
rs = RandomState(MT19937(SeedSequence(st.slider("Seed"))))

st.write("# Regression Output")

POTENTIAL_X_VARS = ["LotArea", "GrLivArea", "YearBuilt", "random"]
X_VARS = []
for v in POTENTIAL_X_VARS:
    if st.checkbox(v, value=True):
        X_VARS.append(v)


y_var = ["SalePrice"]


def get_data() -> pd.DataFrame:
    df = pd.read_csv("regression_output/housing.csv")

    df["random"] = rs.rand(df.shape[0])

    df = df[X_VARS + y_var]

    return df


df = get_data()

rows = st.slider("Num Rows", min_value=1, max_value=len(df), value=len(df))

df = df[0:rows]

st.write("Housing data")
# df

X = df[X_VARS]
y = df[y_var]


f_stastic, p_values = f_classif(X, y)

st.write(f"F-stastic for {X_VARS}")
st.write(f_stastic)
st.write(f"p values for {X_VARS}")
"""
If you repeat this experiment 1,000,000,000 times, this is the percent of the
time that you would expect to randomly show
"""
st.write(p_values)

model = LinearRegression()

model.fit(X, y)

st.write("R^2 Score")
st.write(model.score(X, y))

st.write("Intercept")
model.intercept_
st.write("Coefficients")
model.coef_

y_pred = model.predict(X)

fig, ax = plt.subplots()
ax.scatter(y, y_pred)
ax.scatter(y, y)

st.pyplot(fig)

"""
Obviously this looks non-linear

Maybe plot each variable against price, and see what transformations are required
to make it a linear relationship
"""
