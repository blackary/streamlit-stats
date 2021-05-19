import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_classif

"""
Advanced regression video notes:
a.	Dealing with numerical variables.
b.	Dealing with categorical X variables
c.	Dealing with categorical Y variables

Pricei = B0 + B1Agei + B2Odometeri + Ei

Check the scatter plots to make sure if it’s a linear relationship or nonlinear.
Linear regression is linear in the weights.  We could do something like B2Agei^2, or 
B2(1/odometeri), and this is still linear regression.

Logarithms – a way of scaling skewed data.  Log base 10; natural logarithm (base e).

Try taking Ln of a skewed variable so that it is more of a normal, bell-shaped distribution.  

R2 is not comparable between models with different Y variables.

Create a skewed dataset.  Create best fit.  Take log of dataset, and create another best fit and compare.  
Do an actual vs predicted plot, with transforming the variables in various ways.  
Just do one variable and have a bunch of things you can select.  Include variable a, 
include variable b, transform them in various ways, etc.
"""

st.write(f"## ANOVA Testing")

df = pd.read_csv("../regression_output/housing.csv")
df=pd.get_dummies(df)
cols = df.columns.tolist()
#st.write(cols)


######### SINGLE FEATURE ##########
st.write(f"### Comparing single feature")

selection1 = st.selectbox('Variable to investigate:',cols)

mean1 = df[selection1].mean()
median1 = df[selection1].median()
std1 = df[selection1].std()

trans = st.radio('Transformation',['None','log','sqrt'])
if trans=='None':
    trans_data = df[selection1]
if trans=='log':
    trans_data = np.log(df[selection1])
if trans=='sqrt':
    trans_data = np.sqrt(df[selection1])


# Plot the results
fig, axs = plt.subplots(2, 2)

fig.suptitle('Feature Transforms', fontsize=14, fontweight='bold')
ax=axs[0,0]
ax.hist(df[selection1])
ax.set_xlabel('Predictor Value')
ax.set_ylabel('Count')
ax.set_title(f'Original Histogram \nMean: {mean1:.1f}, Median: {median1:.1f}, Std: {std1:.1f}')
ax.legend()

ax=axs[1,0]
ax.scatter(df[selection1],df['SalePrice'])
ax.set_xlabel('Predictor')
ax.set_ylabel('Target')
ax.set_title(f'Original {selection1} vs SalePrice')
ax.legend()

ax=axs[1,1]
ax.scatter(trans_data,df['SalePrice'])
ax.set_xlabel('Predictor')
ax.set_ylabel('Target')
ax.set_title(f'Transformed {selection1} vs SalePrice')
ax.legend()

ax=axs[0,1]
ax.set_title('Transformed data')
ax.hist(trans_data)
ax.set_xlabel('Predictor Value')
ax.set_ylabel('Count')
plt.tight_layout()

st.pyplot(fig)


######################################################

st.write('# Additional')
x = trans_data #df['LotArea'] #df[columns_to_use]
y = df['SalePrice']
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
ax.set_xlabel('Predictor')
ax.set_ylabel('Target')
ax.set_title('Linear regression fit: single feature (LotArea) vs target')
ax.legend()
st.pyplot(fig)



###### NEW MODEL #######
st.write(f'### Multi-Linear Regression')
x = df[['LotArea','OverallQual','YearBuilt']] #df[columns_to_use]
st.write(x)
model = LinearRegression().fit(x, y)
r2 = model.score(x, y)
st.write('R2:',r2)
b0 = model.intercept_
coeffs = model.coef_
st.write('Intercept:',b0)
st.write('Coefficients:',coeffs)
#yhat_new = b0+b1*x+b2*x+b3*x
yvals = model.predict(x)

# Plot the results
fig, ax = plt.subplots()
ax.scatter(y, yvals)
ax.plot(y,y,'orange',label='Perfect prediction')
ax.set_xlabel('Prediction')
ax.set_ylabel('Target')
ax.set_title('Prediction accuracy of multi-linear regression')
ax.legend()
st.pyplot(fig)