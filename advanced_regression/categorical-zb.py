import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

"""
# Categorical X Variables and Interaction Terms


### Try dropping everything but SaleCondition to see the effects of not dropping the first
"""

df = pd.read_csv("regression_output/housing.csv")

print(df.SaleCondition)
df["SaleConditionCat"] = pd.Categorical(df["SaleCondition"])
print(df.SaleConditionCat)
df["SaleConditionCatInts"] = df["SaleConditionCat"].cat.codes.astype("int")

st.write(df["SaleConditionCat"].unique())

df["LogLotArea"] = np.log(df["LotArea"])
df["LogGrLivArea"] = np.log(df["GrLivArea"])

df["AgeAtSale"] = df["YrSold"] - df["YearBuilt"]

POTENTIAL_X_VARS = ["LogLotArea", "LogGrLivArea", "AgeAtSale", "SaleConditionCatInts"]
X_VARS = []
for v in POTENTIAL_X_VARS:
    if st.checkbox(v, value=True):
        X_VARS.append(v)

drop_first = False

if "SaleConditionCatInts" in X_VARS:
    if st.checkbox("SaleCondition Dummies", value=False):
        X_VARS.remove("SaleConditionCatInts")
        X_VARS += [
            "SaleConditionCat_Abnorml",
            "SaleConditionCat_AdjLand",
            "SaleConditionCat_Alloca",
            "SaleConditionCat_Family",
            "SaleConditionCat_Normal",
            "SaleConditionCat_Partial",
        ]
        drop_first = st.checkbox("Drop first?")
        if drop_first:
            X_VARS.remove("SaleConditionCat_Abnorml")

df = pd.get_dummies(df, columns=["SaleConditionCat"], drop_first=drop_first)

df["LogSalePrice"] = np.log(df["SalePrice"])
y_var = ["LogSalePrice"]

st.write(df[X_VARS + y_var])


X = df[X_VARS]

y = df[y_var]

lr = LinearRegression()
lr.fit(X, y)

st.write("R^2 Score")
st.write(lr.score(X, y))

equation = "Price = "
for i, coef in enumerate(lr.coef_[0]):
    equation += f"{coef:.2f} \* {X_VARS[i]} + "

equation += f"{lr.intercept_[0]:.2f}"

st.write(equation)
