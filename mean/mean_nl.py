import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.title('Mean comparison')
#st.write('Here is a dataframe:')

#"""
## Create the dataset of your choice:
#"""

df = pd.DataFrame({
  'first column': ['Random Normal', 'Uniform 0-1', 'Poisson', 4],
  'second column': [1, 10, 30, 40]
})


# Use a selectbox to choose from a series.
# Put options in the sidebar.
option = st.sidebar.selectbox(
    'Type of data to plot:',
     df['first column'])

'You selected:', option

num_pts = st.sidebar.slider("How many values:", min_value=1, max_value=200, value=20)

# Create the dataset of choice.
if option=='Random Normal':
    data = pd.DataFrame(
        np.random.randn(num_pts, 1),
        columns=['Data'])
elif option=='Uniform 0-1':
    data = pd.DataFrame(
        np.random.rand(num_pts, 1),
        columns=['Data'])
elif option == 'Poisson':
    lam = st.sidebar.selectbox(
        'Select lambda for distribution:',
        df['second column']
    )
    data = pd.DataFrame(
        np.random.poisson(lam,num_pts),
        columns=['Data'])
else:
    data = pd.DataFrame(
        np.random.randn(num_pts, 1),
        columns=['Data'])

# Use checkbox to show/hide mean of data.
# Put option on sidebar.
if st.sidebar.checkbox('Show mean'):
    data['Mean'] = data['Data'].mean()

# Plot the data
st.line_chart(data)
