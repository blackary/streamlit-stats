import numpy as np

"""
Advanced regression video notes: Categorical X variables.
Multi-linear regression for car sale price.   
"""
age = 36
odometer = 290  # Thousand km
pinkslip = 1

# Classify age into categorical variable.
# Note: agecat1 is not used to avoid dummy variable trap.
agecat2,agecat3,agecat4 = 0,0,0
if age>=5 and age<15:
    agecat2=1
elif age>=15 and age<35:
    agecat3=1
elif age>=35:
    agecat4=1

# Apply regression equation (specified in video).
lnprice = ( 9.125-0.181*agecat2-0.8*agecat3-0.39*agecat4
    -0.209*(np.log(odometer))+0.123*pinkslip+1.371*(pinkslip*agecat4) )

price = np.exp(lnprice)
print(f'${price:.2f}')
