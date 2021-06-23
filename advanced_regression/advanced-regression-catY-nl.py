import numpy as np

"""
Advanced regression video notes:  
Categorical Y variables.
"""
""" 
For a yes/no Y, we can predict the probability of each outcome.
We could also predict the odds: = pi/1-pi, [0,inf], midpoint=1. But, this is skewed assymetrical.
Instead, take log(odds), [-inf,inf], midpoint=0. Now, the regression will never create a value of
y outside of its range, and it is symmetric. It is binomial logistic regression.
    This is estimated using maximum likelihood estimation.
Odds Ratio (OR): The multiplicative effect of one extra variable on the odds of selling the car.
    Takes the coefficient and raises it to the power so it's not inside a log (harder to interpret).
Coefficients in equation tell us the effect of the X variable on the log of the odds of the Y variable.

Ex. Probability of rain tomorrow is 20%.  Odds = P/0.2/0.8 = 0.25 = 1:4 (1 in 4).
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
