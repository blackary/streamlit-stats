"""
Kurtosis dashboard experiment.
Originally: describes the "peakedness" of a distribution.
Mean and standard deviation (and skew) are same, but distribution can have higher
peak and slightly higher tails. More peaked and fatter tails = higher kurtosis.

First moment: mean. 2nd centralised moment: variance. 3rd standardised moment: skew.
Fourth (standardised) moment: kurtosis. Population is easy, sample eqn is difficult.

Normal distribution has kurtosis=3. More pronounced central and tails >3.
Broader shoulders, stocky distribution <3.  Ranges from 1 to infinity.
Skinnier shoulders, more kurtosis.  Kurt is rudely brief and blunt, so it falls off 
quickly as you go out.
Excess kurtosis = kurtosis-3.
Recall, standard deviation=variance^2.

Controversy: tails matter a lot more than values near the mean for kurtosis. We refer more
to the thickness/fatness of the tails or the presence of outliers and how far away they are.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

st.title('Kurtosis comparison')

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
    vals=ax.hist(data['Data'], density=True, histtype='stepfilled', alpha=0.2,bins=12)
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
