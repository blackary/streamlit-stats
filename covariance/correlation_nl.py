"""
Correlation dashboard experiment.
First, get deviation from mean for both variables. (x-xbar),(y-ybar).
Then, multiply (x-xbar)(y-ybar). Pos: same sign; neg: diff sign.
COV(x,y)=sigmaxy = sum(xdiff)(ydiff)/(n-1).
Degrees of freedom=n-1.
VAR(x)=sigmax^2 = sum(xi-xbar)^2/(n-1). 
Cov is just Var with 2 variables instead of 1.
CORR(x,y)=rhoxy = sigmaxy/(sigmax*sigmay).  -1<=rho<=1.
If using sample, you need to worry about DoF. If using probabilities,
don't need to worry about DoF, since we are not uncertain.
Expected value: sumproduct of probability and variable.
COV tells us if relationship is positive or negative.
CORR tells us the strength of the relationship.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.title("Correlation-Covariance Comparison")

# ----------------------------------------------------------
# Dataset and plot options.
# ----------------------------------------------------------
num_pts = st.sidebar.slider("How many values:", min_value=1, max_value=200, value=20)
linear_coeff = st.sidebar.slider(
    "Linear Coefficient (y=#x):", value=2.0, min_value=-5.0, max_value=5.0, step=0.25
)
random_coeff = st.sidebar.slider(
    "Magnitude of Random Noise:", value=20, min_value=0, max_value=10
)
exp_coeff = st.sidebar.slider("Raise to power:", value=1, min_value=-5, max_value=5)


# ----------------------------------------------------------
# Create the dataset of choice.
# ----------------------------------------------------------
x = np.linspace(1, 100, num=num_pts)
y = (linear_coeff * x + random_coeff * np.random.normal(size=x.shape)) ** exp_coeff

st.write(f"## Using equations from video:")


# ----------------------------------------------------------
# Run the statistical calculations.
# ----------------------------------------------------------
"""
Covariance: tells us if relationship is positive or negative.  
First, get deviation from mean for both variables. (x-xbar),(y-ybar).
Then, multiply (x-xbar)(y-ybar). Pos: same sign; neg: diff sign.
"""
n = num_pts
xdiff = x - np.mean(x)
ydiff = y - np.mean(y)
sigmaxy = sum(xdiff * ydiff) / (n - 1)
st.write("COV(x,y)=sigmaxy = sum(xdiff*ydiff)/(n-1)")
st.write("COV(x,y)=sigmaxy=", sigmaxy)

"""Cov is just Var with 2 variables instead of 1."""
var_x = sum(xdiff ** 2) / (num_pts - 1)
var_y = sum(ydiff ** 2) / (num_pts - 1)
st.write("VAR(x)=sigmax^2 = sum(xi-xbar)^2/(n-1).")
st.write("var(x)=", var_x)
st.write("var(y)=", var_y)
sigmax = var_x ** 0.5
sigmay = var_y ** 0.5

"""
Correlation tells us the strength of the relationship.  
CORR(x,y): sigmaxy/(sigmax*sigmay). -1<=rho<=1.
"""
corr = sigmaxy / (sigmax * sigmay)
st.write("CORR(x,y)=rho=", corr)
st.write(np.cov(x, y))

st.write(f"## Using built-in numpy calculations:")
stacked = np.stack([x, y], axis=0)
st.write(f"Covariance: {np.cov(stacked)[0][1]}")
st.write(f"Correlation: {np.corrcoef(stacked)[0][1]}")


# ----------------------------------------------------------
# Plot the data.
# ----------------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(x, y)
st.pyplot(fig)
