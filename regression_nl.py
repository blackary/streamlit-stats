"""
Regresssion
Most simply, a line of best fit. Yhat = intercept+slope*X. (sample regression line.)
Yhat line = Min(sum(error^2)), where error=e=point-line.
Minimize sum of squared error, or sum of abs(error)?  Why?
Ybar = mean value of Y (ie, bar takings).
Yhat line shows what is expected value of Y for given value of X, based
on relationship.
Yhati-Ybar gives expected/explained deviation from mean for Xi.
ei = Yi-Yhati is unexplained deviation from mean (error).
SSR = sum(Yhati-Ybar)^2 - explained components
SSE = sum(Yi-Yhati)^2 - unexplained components
SST = SSR+SSE= sum(Yi-ybar)^2
R2 = proportion SSR/SST of total sum of squares.  Higher R^2, lower SSE, 
more of deviation from mean is explained by the Yhat regression line.
Population (true) regression function (which we don't know):
Y=B0+B1X+e_curly
"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import scipy.stats

st.title("Regression Analysis")

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


# ----------------------------------------------------------
# Run the statistical calculations.
# ----------------------------------------------------------
#Yhat = slope*x + intercept = mx+b # Regression line/Line of best fit.
st.write('### Manually calculate the R2 value:')
"""
1. Fit the regression line to the data: yhat = slope*x + intercept
2. Unexplained deviation from the mean (error of fit): ei = y-yhat
3. Calculate SSR: portion of variance explained by regression.
SSR = sum((yhat-ybar)^2)
4. Calculate SSE: portion of variance not explained by regression.
SSR = sum((yhat-ybar)^2)
5. Calculate SST and R^2: proportion SSR/SST of total sum of squares.  Higher R^2, lower SSE, 
more of deviation from mean is explained by the Yhat regression line.
6. Calculate R^2: proportion SSR/SST of total sum of squares.  Higher R^2, lower SSE, 
more of deviation from mean is explained by the Yhat regression line.
"""
m,b = np.polyfit(x,y,1)  # Fit the first-order regression to the data.
yhat = m*x+b

# Step 2
ei = y-yhat
ybar = np.mean(y)
# Calculate error of fit.

#Step 3
SSR = sum((yhat-ybar)**2)
st.write(f'SSR: {SSR:.2f}')

#Step 4
SSE = sum((ei)**2)
st.write(f'SSE: {SSE:.2f}')

#Step 5
SST = SSR+SSE
st.write(f'SST: {SST:.2f}')

#Step 6
R2 = SSR/SST
st.write(f'R2: {R2:.2f}')
eqn1 = f'y={m:.2f}x+{b:.2f}'

# Plot the results
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x,yhat,'r',label=eqn1)
ax.set_title('Plot of yhat (manual calculation)')
ax.legend()
st.pyplot(fig)

# Automatic way
st.write('### Now, do all these steps automatically using scipy.stats.linregress(x,y):')

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
yhat_2 = slope*x + intercept
R2_2 = r_value**2
st.write(f'R2: {R2_2:.2f}')
eqn2 = f'y={slope:.2f}x+{intercept:.2f}'

# Plot results
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x,yhat_2,'r',label=eqn2)
ax.set_title('Plot of yhat (from scipy linregress)')
ax.legend()
st.pyplot(fig)