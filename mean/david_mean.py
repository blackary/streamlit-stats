import streamlit as st
import numpy as np
from matplotlib import pyplot as plt


dist_type = st.sidebar.radio("Distribution", ["normal", "uniform", "lognormal"])

N = st.sidebar.slider("N", 2, 500, 10, 1)

NUM_BINS = int(np.ceil(np.sqrt(N)))

if dist_type == "normal":
    MIN_MEAN = -1.0
    MAX_MEAN = 1.0
    MIN_SD = 0.0
    MAX_SD = 5.0
    mn = st.sidebar.slider("mean", MIN_MEAN, MAX_MEAN, 0.0, 0.1)
    sd = st.sidebar.slider("sd", MIN_SD, MAX_SD, 1.0, step=0.1)
    X = np.random.randn(N) * sd + mn
    min_x = MIN_MEAN - MAX_SD * 2
    max_x = MAX_MEAN + MAX_SD * 2
elif dist_type == "uniform":
    mn = 0.5
    MIN_MEAN = 0.5
    MAX_MEAN = 0.5
    MIN_SD = 0.5
    MAX_SD = 0.5
    X = np.random.uniform(0, 1, N)
    min_x = 0
    max_x = 1
elif dist_type == "lognormal":
    MIN_MEAN = -1.0
    MAX_MEAN = 1.0
    MIN_SD = 0.0
    MAX_SD = 5.0
    mu = st.sidebar.slider("mu", MIN_MEAN, MAX_MEAN, 0.0, 0.1)
    sd = st.sidebar.slider("sd", MIN_SD, MAX_SD, 1.0, step=0.1)
    mn = np.exp(mu + sd ** 2 / 2)
    X = np.exp(np.random.randn(N) * sd + mu)
    min_x = MIN_MEAN - MAX_SD * 2
    max_x = MAX_MEAN + MAX_SD * 2

fig, ax = plt.subplots()
heights, bins, hist_fig = ax.hist(X, bins=int(np.ceil(np.sqrt(N))))

sample_mean = np.mean(X)
sample_sd = np.sqrt(np.var(X, ddof=1))

max_height = int(np.ceil(max(heights) / 100) * 100)
max_x = int(np.ceil(max(bins) / 20) * 20)

ax.vlines(sample_mean, 0, max_height, color="red", label="Sample Mean")
ax.vlines(mn, 0, max_height, color="green", label="Distribution Mean")
ax.set_xlim([min_x, max_x])
ax.set_ylim([0, max_height])
ax.legend()

st.write(fig)
