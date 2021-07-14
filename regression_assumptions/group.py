import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.gofplots import qqplot

"# What happens if the regression assumptions are violated?"


if st.sidebar.checkbox("Fixed seed?", value=True):
    np.random.seed(42)


def underlying_function(x: np.ndarray) -> np.ndarray:
    return x * 4 + 12


stdev = st.sidebar.slider("Std Dev", 0, 10, 2)


def add_error(x: np.ndarray) -> np.ndarray:
    dist_type = st.sidebar.selectbox(
        "Distribution type", ["normal", "poisson", "uniform", "log_normal"]
    )
    if dist_type == "normal":
        noise = np.random.normal(scale=stdev, size=x.shape)
    elif dist_type == "poisson":
        lam = stdev ** 2
        noise = np.random.poisson(lam=lam, size=x.shape)
        noise -= lam
    elif dist_type == "uniform":
        b = np.sqrt((stdev ** 2) * 6)
        noise = np.random.uniform(low=-b, high=b, size=x.shape)
    elif dist_type == "log_normal":
        # THIS STDEV PROBABLY WRONG
        noise = np.random.lognormal(sigma=np.log(stdev), size=x.shape)
    else:
        raise ValueError(dist_type)

    return x + noise


num_points = st.sidebar.slider("Num points", 0, 100, 30)

X = np.arange(0, num_points, 1)

y_ = underlying_function(X)
y = add_error(y_)

reg = LinearRegression()
reg.fit(X.reshape(-1, 1), y)
m = reg.coef_[0]
b = reg.intercept_

y_pred = m * X + b

y_res = y_pred - y

fig, ax = plt.subplots()

ax.plot(X, y_, color="red", label="underlying")
ax.scatter(X, y, color="blue", label="noisy")
ax.plot(X, y_pred, color="green", label="predicted")

ax.legend()

st.write(fig)


fig, ax = plt.subplots()

# ax.scatter(y, y_res, color="red", label="Residual")

# ax.set_title("Histogram of residuals")
ax.hist(y_res, label="residuals")

ax.legend()

st.write(fig)

fig, ax = plt.subplots()

ax.set_title("Q-Q Plot")
qqplot(y_res, line="s", ax=ax)

st.write(fig)
