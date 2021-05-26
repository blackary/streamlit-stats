import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""
Advanced regression video notes:
Categorical X variables.
Experiment: multi-level categorical variables.
    a. Split car age into 4 categories at 5, 15, and 35.
        - Warning: This implies a linear, consistent increase in price
        between each category, which may not be true.
    c. Instead, split into 4 different binary categorical variables.
        - AgeCat1, AgeCat2, AgeCat3, AgeCat4.
        - Dummy variable TRAP: We can't include all 4 variables into our model,
        because AgeCat1 can be completely explained by the other 3 classes.
        - SHOW THIS: compare linear model with all 4 and model with only 3.
        - Base case is the least interesting - it's the one category you remove.
            - All other categories' impacts on model training are relative to base case.
            ie a car in age cat 4 will command price 47% higher than car in age cat 1.
Interaction terms: required when X1 affects the relationship btwn X2 and Y.
eg Age of employee and uni degree. Basically an Effect modifier - a degree modifies the 
impact of age on salary.
    - ie B6(Pink slip)*(AgeCat4). A vintage car that is ALSO road-worthy is additionally valuable.
"""

def fit_linreg(x,y):
    """
    Fit the first-order regression to the data.
    Returns m, b, yhat, R2.
    """
    m,b = np.polyfit(x,y,1)
    yhat = m*x+b

    # Step 2
    ei = y-yhat
    ybar = np.mean(y)
    # Calculate error of fit.
    #Step 3
    SSR = sum((yhat-ybar)**2)
    #Step 4
    SSE = sum((ei)**2)
    #Step 5
    SST = SSR+SSE
    #Step 6
    R2 = SSR/SST

    return m,b,yhat,R2


st.write(f"# ANOVA Testing")

df = pd.read_csv("regression_output/housing.csv")
df = pd.get_dummies(df)
cols = df.columns.tolist()


# ---------------------------------------------------------------
# COMPARE ORIGINAL FEATURE WITH TRANSFORMED FEATURE.
# ---------------------------------------------------------------
st.write(f"## Comparing single feature")
"""Analyze how different transformations impact the data distribution."""

selection1 = st.selectbox('Variable to investigate:',cols)

orig_data = df[selection1]
mean1 = orig_data.mean()
median1 = orig_data.median()
std1 = orig_data.std()

trans = st.radio('Transformation',['None','log','sqrt'])
if trans=='None':
    trans_data = orig_data
if trans=='log':
    trans_data = np.log(orig_data)
if trans=='sqrt':
    trans_data = np.sqrt(orig_data)

mean2 = trans_data.mean()
median2 = trans_data.median()
std2 = trans_data.std()


# ---------------------------------------------------------------
# # Plot the feature comparison results.
# ---------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(10,10))

fig.suptitle('Feature Transforms', fontsize=14, fontweight='bold')
ax=axs[0,0]
ax.hist(df[selection1])
ax.set_xlabel('Predictor Value')
ax.set_ylabel('Count')
ax.set_title(f'Histogram - Original Data \nMean: {mean1:.1f}, Median: {median1:.1f}, Std: {std1:.1f}')

ax=axs[1,0]
ax.scatter(df[selection1],df['SalePrice'])
ax.set_xlabel('Predictor')
ax.set_ylabel('Target')
ax.set_title(f'Original {selection1} vs SalePrice')

ax=axs[0,1]
ax.hist(trans_data)
ax.set_xlabel('Predictor Value')
ax.set_ylabel('Count')
ax.set_title(f'Histogram - Transformed Data \nMean: {mean2:.1f}, Median: {median2:.1f}, Std: {std2:.1f}')

ax=axs[1,1]
ax.scatter(trans_data,df['SalePrice'])
ax.set_xlabel('Predictor')
ax.set_ylabel('Target')
ax.set_title(f'Transformed {selection1} vs SalePrice')

plt.tight_layout()
st.pyplot(fig)


# ---------------------------------------------------------------
# CREATE SINGLE LINEAR REGRESSION MODEL USING BOTH DATASETS.
# ---------------------------------------------------------------
st.write('## Create Linear Regression')
"""Compare linear regression with original vs transformed input data."""
x = orig_data
xt = trans_data #df['LotArea'] #df[columns_to_use]
y = df['SalePrice']

m,b,yhat_orig,R2_orig = fit_linreg(x,y)
eqn_orig = f'y={m:.2f}x+{b:.2f}'

m,b,yhat_new,R2_new = fit_linreg(xt,y)
eqn_new = f'y={m:.2f}x+{b:.2f}'

st.write(f'R2 Original: {R2_orig:.2f}')
st.write(f'R2 New: {R2_new:.2f}')


# ---------------------------------------------------------------
# Plot the linear regression results.
# ---------------------------------------------------------------
fig, axs = plt.subplots(1, 2,figsize=(10,5))

fig.suptitle('Linear Regression Fit - single feature vs target (sale price)', fontsize=14, fontweight='bold')
ax=axs[0]
ax.scatter(x, y)
ax.plot(x,yhat_orig,'r',label=eqn_orig)
ax.set_xlabel('Predictor Value')
ax.set_ylabel('Count')
ax.set_title(f'Using Original Input')
ax.legend()

ax=axs[1]
ax.scatter(xt, y)
ax.plot(xt,yhat_new,'r',label=eqn_new)
ax.set_xlabel('Predictor')
ax.set_ylabel('Target')
ax.set_title(f'Using Transformed Input')
ax.legend()
plt.tight_layout()
st.pyplot(fig)


# ---------------------------------------------------------------
# CREATE MULTI-LINEAR REGRESSION MODEL.
# ---------------------------------------------------------------
st.write(f'## Multi-Linear Regression')
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