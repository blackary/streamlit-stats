import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.write(f"# Categorical Features - Regression")
"""
Advanced regression video notes:  
Categorical X variables.  Experiment: multi-level categorical variables.  
"""
""" 
1. Split car age into 4 categories at 5, 15, and 35.  
   - Warning: This implies a linear, consistent increase in price between each category,
   which may not be true.  
   - Instead, split into 4 different binary categorical variables.  
      - AgeCat1, AgeCat2, AgeCat3, AgeCat4.
   - Dummy variable TRAP: We can't include all 4 variables into our model, 
   because AgeCat1 can be completely explained by the other 3 classes.
      - SHOW THIS: compare linear model with all 4 and model with only 3.  
      - Base case is the least interesting - it's the one category you remove.
         - All other categories' impacts on model training are relative to base case.
            ie a car in age cat 4 will command price 47% higher than car in age cat 1.  
2. Interaction terms: required when X1 affects the relationship btwn X2 and Y.  
   - eg Age of employee and uni degree. Basically an Effect modifier - a degree modifies the 
impact of age on salary.  
   - ie B6(Pink slip)*(AgeCat4). A vintage car that is ALSO road-worthy is additionally valuable. If dummy
   columns are a linear combination fo each other, that's when you really need to drop the first dummy.
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


# Divide a variable into categories.
# Test the dummy variable trap.
# ---------------------------------------------------------------
# COMPARE ORIGINAL FEATURE WITH TRANSFORMED FEATURE.
# ---------------------------------------------------------------
st.write("## Analyze how the dummy variable trap impacts model fitting.")
df = pd.read_csv("regression_output/housing.csv")

feature_type = st.sidebar.selectbox('Type of feature:',['string','number'])
if feature_type=='number':
    cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
else:
    cols = df.select_dtypes(include=['object']).columns.tolist()
#st.write(cols)
selection1 = st.sidebar.selectbox('Feature to investigate:',cols)

orig_data = df[selection1]
st.write('Original feature unique values:')
st.write(orig_data.unique())  #SaleCondition, SaleType

# Get dummies splits into multiple 1/0 categorical columns.
# pd.Categorical creates single categorical column (must then convert to ints).
drop_first = st.sidebar.radio('Drop first dummy',['True','False'])
if drop_first=='True':
    dummies = pd.get_dummies(orig_data,drop_first=True)
else:
    dummies = pd.get_dummies(orig_data,drop_first=False)
st.write('Convert to dummies:')
st.write(dummies)


# ---------------------------------------------------------------
# CREATE MULTI-LINEAR REGRESSION MODEL.
# ---------------------------------------------------------------
st.write(f'## Multi-Linear Regression')
x = dummies
y = df['SalePrice']

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