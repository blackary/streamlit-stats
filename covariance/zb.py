import streamlit as st
import numpy as np

# import random
from matplotlib import pyplot as plt

st.write("# Covariance and Correlation")

np.random.seed(st.number_input("Random seed", value=42))

num_points = st.slider("Number of points", value=100, min_value=10, max_value=1000)

x = np.linspace(1, 100, num=num_points)

linear_coeff = st.slider(
    "Linear Coeff", value=2.0, min_value=-5.0, max_value=5.0, step=0.25
)
random_coeff = st.slider("Random Coeff", value=10, min_value=-10, max_value=10)
sin_coeff = st.slider("Sin Coeff", value=10, min_value=-10, max_value=10)
y = (
    linear_coeff * x
    + random_coeff * np.random.normal(size=x.shape)
    + sin_coeff * np.sin(x)
)

st.write(
    f"## y = {linear_coeff} \* x + {random_coeff} \* random() + {sin_coeff} \* sin(x)"
)

fig, ax = plt.subplots()

ax.scatter(x, y)

st.pyplot(fig)

combined = np.stack([x, y], axis=0)

# combined

st.write(f"### Covariance: {np.cov(combined)[0][1]}")
st.write(f"### Correlation: {np.corrcoef(combined)[0][1]}")

"# Does Covariance ~ 0 necessarily imply independence?"

x2 = np.linspace(-100, 100, num=num_points)
y2 = np.piecewise(x2, [x2 < 0, x2 >= 0], [lambda x: -x, lambda x: x])

fig2, ax2 = plt.subplots()

ax2.scatter(x2, y2)

st.pyplot(fig2)

combined2 = np.stack([x2, y2], axis=0)

# combined

st.write(f"### Covariance: {np.cov(combined2)[0][1]}")
st.write(f"### Correlation: {np.corrcoef(combined2)[0][1]}")
