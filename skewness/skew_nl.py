"""
Interactive dashboard for visualizing skew of dataset.
Skew of 1 means mean is separated from median by 1 standard deviation.
Right (positive) skew has a tail to the right; mode<median<mean.
Physics:
Velocity - acceleration - jerk - jounce(snap) - crackle - pop.
Car driving - slowly apply brake - slam on brake - car slams into wall.
Moments of mean: Difficulty to Lift shake spin.

First moment: sum(x)/n = mean mu.
Second (centralized) moment: sum(x-mu)^2/n = variance sigma^2.
Third (standardized) moment: sum(x-mu)^3/(n*sigma^3) = skew.
Just use the built-in functions. These above are for full population, not a sample.

Highly skewed = greater than +/-1. Mod skew: btwn 0.5 and 1.  Not skewed: between -0.5 to 0.5.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

st.title('Skewness comparison')

# ----------------------------------------------------------
# Dataset and plot options.
# ----------------------------------------------------------
df = pd.DataFrame({
  'distribution': ['Random Normal', 'Uniform 0-1', 'Poisson', 'Linear', 'Skewed'],
  'lambda': [1, 10, 20, 30, 40],
})

option = st.sidebar.selectbox(
    'Type of data to plot:',
     df['distribution'])
'You selected:', option

plot_type = st.sidebar.selectbox(
    'Type of data to plot:',
     ['Line plot','Bar plot'])
'You selected:', plot_type

num_pts = st.sidebar.slider("How many values:", min_value=1, max_value=200, value=20)

# ----------------------------------------------------------
# Create the dataset of choice.
# ----------------------------------------------------------
if option == 'Random Normal':
    data = pd.DataFrame(
        np.random.randn(num_pts, 1),
        columns=['Data'])
elif option == 'Uniform 0-1':
    data = pd.DataFrame(
        np.random.rand(num_pts, 1),
        columns=['Data'])
elif option == 'Poisson':
    lam = st.sidebar.selectbox(
        'Select lambda for distribution:',
        df['lambda'])
    data = pd.DataFrame(
        np.random.poisson(lam,num_pts),
        columns=['Data'])
elif option == 'Linear':
    x = np.arange(num_pts)
    delta = np.random.uniform(-5,5, size=(num_pts,))
    data = pd.DataFrame(
        .4 * x + delta,
        columns=['Data'])
elif option == 'Skewed':
    skew_param = st.sidebar.slider("Skewness parameter:", min_value=-100, max_value=100, value=20)
    data = pd.DataFrame(
        stats.skewnorm.rvs(skew_param, size=num_pts),
        columns=['Data'])
else:
    data = pd.DataFrame(
        np.random.randn(num_pts, 1),
        columns=['Data'])

data['Mean'] = data['Data'].mean()
data['Median'] = data['Data'].median()

# ----------------------------------------------------------
# Plot the data.
# ----------------------------------------------------------
if plot_type == 'Bar plot':
    fig, ax = plt.subplots()
    vals=ax.hist(data['Data'], density=True, histtype='stepfilled', alpha=0.2,bins=14)
    st.pyplot(fig)
    n=vals[0]
    bins=vals[1]
    loc = np.where(n==np.max(n))[0]
    mode = (bins[loc]+bins[loc+1])/2

    st.write(f"Mean: {data['Mean'].values[0]:.2f}, Median: {data['Median'].values[0]:.2f}, Mode: {mode[0]:.2f}")
else:
   st.line_chart(data)
   st.write(f"Mean: {data['Mean'].values[0]:.2f}, Median: {data['Median'].values[0]:.2f}")

# Calculate Skewness.
if st.sidebar.checkbox('Show skewness'):
    skew = stats.skew(data['Data'])
    st.write(f"Skewness: {skew:.2f}")
