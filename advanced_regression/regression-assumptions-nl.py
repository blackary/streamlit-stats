import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""
Regression Assumptions
"""
""" 
Coefficients can be biased.  Standard errors could be unreliable (then so too would be t-stat and p-value).
1. Linearity
    The regression needs to be linear in the parameters. You can have x^2 terms to represent squared data, so
    the relationship is linear with y (and additive).
2. Constant Error Variance
    If you map out the distances between points and black fitted line, the variance scales with the values of the data.
    We don't want Heteroskedasticity.
3. Independent Error Terms
    Autocorrelation - a snaking plot - is bad.
4. Normal errors
    Error spread needs to be normally distributed (bell-shaped).
5. No multi-collinearity
    Need to have truly independent x terms.
6. Exogeneity
    Omitted variable bias.
"""
"""
Detailed 1:
Residuals show the error in each observation relative to the fitted regression.
We want to see a nice even spread.  y = b0 + b1(age)i + b2(age^2)i + e
This is still a linear regression, because the inputs are linear wrt y.
DO: residual plots, fit a linear regression to squared relation, and don't account for it.
Estimated coefficients are not the best possible.

Detailed 2:
The spread in residuals increases as x increases (variance increases with x) - bad.
Standard errors cannot be relied upon, but fitted result is still reasonably accurate.
You can try to fix this by logging both variables.

Detailed 3:
This can only occur with ordered/time-series data. Wiggly residuals. Each residual is affected by the one before it.
Estimated coefficients are still unbiased, but standard errors in output cannot be relied upon.
Remedies: Investigate omitted variables (ie a business cycle for a stock index); generalised difference equation.

Detailed 4:
If there are a ton of zeros and a few other points, the standard errors are affected (if n is small).
The true relationship will come out when there are enough examples. Small number of obs, standard errors are affected.
Detect using a Q-Q plot. (If obs are on a sttraight diagonal line, it is normally-distributed.)
To fix: change functional form (log?) and it's not a big deal just get more obs.

Detailed 5:
Number of cars and number of residents in development are pretty related (multi-collinearity).
When we interpret each coefficient in a linear regression, we are "holding all other variables constant".
If this isn't possible, then the model can't tell which of these actually made the difference - can't separate the effects.
Coefficients and standard errors of affected variables are unreliable. To find: look at correlation (p) btwn X variables.
Variance Inflation Factor (VIF) shows how much that variable's information is hidden in other, existing variables in model.
Solution: remove one of variables. DO NOT multiply the variables together to fix this.

Detailed 6:
Omitted variable bias - a variable affects both X and Y, but is not included in the model.
Socio-economic status would affect error term ei in the model. Thus, education is no longer wholly exogenous as it can be explained
in part by the error term. Error term is not fully random. Children of professors vs children of janitors - this likely impacts
years of education and salary. Result: Model can only be used for predictive purposes (can not infer causation). Can still get a
valid prediction, but I can't say it's that 13th year of education that gave me this amount of higher salary.
"""

def fit_linreg(x: np.ndarray, y: np.ndarray):
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


def single_regression(x,y):
    st.write(f"## Linear Regression")
    m,b,yhat,R2 = fit_linreg(x,y)
    eqn1 = f'y = {m:.1f}x+{b:.1f}'

    # Plot the results of linear fit
    fig, ax = plt.subplots()
    ax.scatter(x, y,label='Feature Data')
    ax.plot(x,yhat,'r',label=eqn1)
    ax.set_title('Plot of Linear Fit')
    ax.legend()
    st.pyplot(fig)
    st.write(f"#### R2: {R2:.4f}")

    # Plot the RESIDUALS
    fig, ax = plt.subplots()
    ax.scatter(x, yhat-y, color='black')
    #ax.plot(x,yhat-y,'r',label=eqn1)
    ax.set_title('Plot of Residuals')
    st.pyplot(fig)
    #sns.residplot(x=x, y=y, lowess=True, color="g")


def multi_regression(x_mult,y):
    st.write(f'## Multi-Linear Regression')
    st.write(f'### Fitted Regression:')
    model = LinearRegression().fit(x_mult, y)
    r2 = model.score(x_mult, y)
    yvals = model.predict(x_mult)

    # Plot the results
    fig, ax = plt.subplots()
    ax.scatter(y, yvals, label='Model prediction')
    ax.plot(y,y,'orange',label='Perfect prediction')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Target')
    ax.set_title('Prediction accuracy of multi-linear regression')
    ax.legend()
    st.pyplot(fig)

    b0 = model.intercept_
    coeffs = model.coef_
    st.write(f"Intercept: {b0:.2f}")
    st.write('Coefficients:',coeffs)
    st.write(f"#### R2: {r2:.3f}")


def run_vif(x: pd.DataFrame):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    st.write(f"### Feature Relationships:")
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = x.columns
    # Calculating VIF for each feature
    names = x.columns
    vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(names))]
    vif_data = vif_data.style.background_gradient(cmap='Reds',vmin=0,vmax=100)
    corr_table = x.corr().style.background_gradient(cmap='bwr',vmin=-1,vmax=1)
    st.write('Correlation:',corr_table)
    st.write('VIF:',vif_data)


# ---------------------------------------------------------------
# READ AND SELECT DATA OF INTEREST.
# ---------------------------------------------------------------
df = pd.read_csv("regression_output/housing.csv")
feature_type = st.sidebar.selectbox('Type of feature:',['string','number'])

if feature_type == 'number':
    cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    selections = st.sidebar.multiselect('Select one or more features:', cols)
    chosen_data = df[selections]
    st.write(f"### Chosen data:")
    st.write(chosen_data)
elif feature_type == 'string':
    cols = df.select_dtypes(include=['object']).columns.tolist()
    selections = st.sidebar.selectbox('Select a feature:', cols)

    orig_data = df[selections]
    st.write(f'#### Original feature unique values:')
    st.write(orig_data.unique())  #SaleCondition, SaleType

    # Get dummies splits into multiple 1/0 categorical columns.
    # pd.Categorical creates single categorical column (must then convert to ints).
    drop_first = st.sidebar.radio('Drop first dummy',['True','False'])
    if drop_first == 'True':
        chosen_data = pd.get_dummies(orig_data,drop_first=True)
    else:
        chosen_data = pd.get_dummies(orig_data,drop_first=False)
    st.write(f'#### Convert to dummies:')
    st.write(chosen_data)


# ----------------------------------------------------------
# Create the dataset of choice.
# ----------------------------------------------------------
x = chosen_data
y = df['SalePrice']

if len(chosen_data.columns) > 1:
    multi_regression(x,y)
    run_vif(x)
elif len(chosen_data.columns) == 1:
    x = x.to_numpy()[:,0]
    single_regression(x,y)
