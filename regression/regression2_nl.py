"""
Regresssion
Drawing a line of best fit through the observations.
Line goes through every observation, R2==1.
Further away observations travel from line of best fit, R2 decreases.
    This means the relationship is weaker, x is explaining less of variance.
1 is perfect linear relationship. 0 has no linear component.
Degrees of freedom:
To make a linear regression, you can draw a line btwn 2 points, but to assess the 
strength of the relationship, you need a third point. This is 1 degree of freedom.
Ex: with 2 explanatory variables (X1,X2)->Y, we need 3 points to fit a plane.
We need a 4th point to get 1 degree of freedom, to assess the strength of relationship.

df = n-k-1. n=#obs, k=order of relationship fit (number of explanatory X variables).
As df decreases (ie add more variables to model), R2 will ONLY increase. If you make your
model more powerful, it will fit your data better. If useless, could just be 0 coefficient on that variable.

Adjusted R2. Account for uncertainty of R2 when df is lower. As k increases, Adj R2 decreases, 
reflecting reduced ability of model to obtain statistically-meaningful results.
If lots of observations, more variables may, in fact, be helpful for model accuracy.
If only a few observations, more variables may make model worse. Relative explanatory power.
If Adj R2 is higher, does that mean that model is Always better than the other?
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