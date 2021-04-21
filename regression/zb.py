import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.optimize import minimize

st.title("Regression -- Sum of Square Errors vs Sum of Absolute Errors")

seed = st.slider("Seed", min_value=0, max_value=100, value=42)
np.random.seed(seed)


noise = st.slider("Noise", min_value=0, max_value=100, value=5)

noise_type = st.selectbox("Noise type", ["Normal", "F"])


def model(x):
    return 2 * x + 5


def model_with_noise(x):
    if noise_type == "Normal":
        return model(x) + np.random.normal(scale=noise, size=x.shape)
    elif noise_type == "F":
        return model(x) + np.random.noncentral_f(3, 20, noise * 2, size=x.shape)


def sum_abs_error(params, X, Y):
    slope = params[0]
    intercept = params[1]
    Y_hat = X * slope + intercept
    return np.sum(np.abs(Y_hat - Y))


def sum_squared_error(params, X, Y):
    slope = params[0]
    intercept = params[1]
    Y_hat = X * slope + intercept
    return np.sum((Y_hat - Y) ** 10)


arr = np.array([1, 2, 3])

X = np.linspace(0, 100, num=1000)
Y = model_with_noise(X)

fig, ax = plt.subplots()
ax.scatter(x=X, y=Y)

# initial guess for slope and intercept
x0 = np.array([1, 0])

minimized = minimize(sum_abs_error, x0, method="Nelder-Mead", args=(X, Y))
slope = minimized["x"][0]
intercept = minimized["x"][1]

st.write("Underlying model: y = 2 x + 5")
st.write(f"Minimizing sum of absolute error: y = {slope:.3} x + {intercept:.3}")
# st.write(minimized)

ax.plot(X, model(X), "k--", color="purple", label="Underlying model")

ax.plot(X, X * slope + intercept, color="red", label="Minimizing absolute error")
# st.write(minimized)

minimized = minimize(sum_squared_error, x0, method="Nelder-Mead", args=(X, Y))
# st.write(minimized["x"])

slope = minimized["x"][0]
intercept = minimized["x"][1]
st.write(f"Minimizing sum of square error: y = {slope:.3} x + {intercept:.3}")

ax.plot(X, X * slope + intercept, color="green", label="Minimizing square error")
ax.legend()

st.write(fig)

"## Trying taking 100 different samples of X & Y, and doing each fit, and looking at mean & stdev: "
linear_slopes = []
linear_intercepts = []
square_slopes = []
square_intercepts = []
for i in range(100):
    np.random.seed(i)
    X_sub = np.random.choice(X, size=100)
    Y_sub = model_with_noise(X_sub)
    minimized = minimize(sum_abs_error, x0, method="Nelder-Mead", args=(X_sub, Y_sub))[
        "x"
    ]
    linear_slopes.append(minimized[0])
    linear_intercepts.append(minimized[1])
    minimized = minimize(
        sum_squared_error, x0, method="Nelder-Mead", args=(X_sub, Y_sub)
    )["x"]
    square_slopes.append(minimized[0])
    square_intercepts.append(minimized[1])

f"""
Equation using absolute error:

    y = ({np.mean(linear_slopes):.2f} +/- {np.std(linear_slopes):.2f}) x + ({np.mean(linear_intercepts):.2f} +/- {np.std(linear_intercepts):.2f})

Equation using square error:

    y = ({np.mean(square_slopes):.2f} +/- {np.std(square_slopes):.2f}) x + ({np.mean(square_intercepts):.2f} +/- {np.std(square_intercepts):.2f})
"""
