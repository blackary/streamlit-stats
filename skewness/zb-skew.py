import streamlit as st

# from scipy.stats import mode
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# from statsmodels.distributions.empirical_distribution import ECDF

fig, ax = plt.subplots(1, 1)

"# Skewness!"

np.random.seed(42)

s = np.random.normal(0, 1, 100_000)
# s = np.random.lognormal(mean=5, sigma=1, size=100_000)

shifty = st.slider("shifty", min_value=-20, max_value=20, value=10)

s = s * (s + shifty)

# st.write(density, np.argmax(density))
# st.write(weights)

mmin, mmax = min(s), max(s)

# ecdf = ECDF(s)
# s = ecdf(np.linspace(mmin, mmax, num=100))

mean = np.mean(s)
median = np.median(s)
# mode = stats.mode(s).mode[0]
density, weights = np.histogram(s, bins=100)
# ax.set_yscale("log")
mode = weights[np.argmax(density)]
stdev = np.std(s)

count, bins, ignored = ax.hist(s, 100, density=True)

f"## Mean {mean}"
f"## Median {median}"
f"## Mode {mode}"
f"## Standard Deviation {stdev}"

f"### (Mean - Mode) / stdev = {(mean - mode ) / stdev}"
f"### 3(Mean - Median) / stdev = {3*(mean - median) / stdev}"
f"### Mode = 3(Median) - 2(Mean)? {mode} = {3*median} - {2*mean} = {3*median - 2*mean}"

st.pyplot(fig)
