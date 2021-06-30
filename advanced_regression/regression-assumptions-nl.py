import numpy as np

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

"""
# In-video example:

# (a) Probability that a $4,500 car with a pink slip will sell.
slip = 1
price = 4.5  # In thousands
log_odds = 0.396 - 0.173*price + 1.555*slip
odds1 = np.exp(log_odds)
prob1 = odds1/(1+odds1)

# (b) Probability that a $4,500 car without a pink slip will sell.
slip = 0
price = 4.5  # In thousands
log_odds = 0.396 - 0.173*price + 1.555*slip
odds0 = np.exp(log_odds)
prob0 = odds0/(1+odds0)

# (c) Odds ratio for odds of sale (a) vs odds of sale (b).
# Thus, a number greater than 1 means a car with a pink slip is more
# likely to sell than a car without a pink slip.
odds_ratio = odds1/odds0
print(prob1)
print(prob0)
print(odds_ratio)
