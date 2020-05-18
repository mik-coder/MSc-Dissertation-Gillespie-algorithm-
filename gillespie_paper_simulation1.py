# https://github.com/karinsasaki/gillespie-algorithm-python/blob/master/build_your_own_gillespie_exercises.ipynb
# ^^^ Github example of stochastic implementation ^^^
# http://be150.caltech.edu/2016/handouts/gillespie_simulation.html
# ^^^ Example algorithm ^^^

# Implement a tau-leaping variant of the SSA   

import numpy as np
import scipy
from scipy.special import binom
from scipy import stats
from matplotlib import pyplot as plt # allow inline plot with %matplotlib inline

# tau-leaping SSA variant of an Irriversible Isomerisam process!  

# System of equations
""" S1 --> 0 == 1 --> 0
"""

# Need to initialise the discrete numbers of molecules???
popul_num = np.array([1.0E5])
# should use --> [1.0E5]

# ratios of starting materials for each reaction 
LHS = np.array([1])

# ratios of products for each reaction
RHS = np.matrix([0])

# stochastic rates of reaction
stoch_rate = np.array([1.0])

# Define the state change vector
state_change_array = np.asarray(RHS - LHS)

# Intitalise time variables
tmax = 20.0         # Maximum time
tao = 0.0           # array to store the time of the reaction.


# function to calcualte the propensity functions for each reaction
def propensity_calc(LHS, popul_num, stoch_rate):
    propensity = np.zeros(len(LHS))
    for row in range(len(LHS)):
            a = stoch_rate[row]     
            for i in range(len(popul_num)):
                if (popul_num[i] >= LHS[row]):       
                    binom_rxn = binom(popul_num[i], LHS[row])
                    a = a*binom_rxn
                else:
                    a = 0
                    break
            propensity[row] = a     
    return propensity

print(propensity_calc(LHS, popul_num, stoch_rate))



# Tau-leaping  method

delta_t = 0.1
# epsi --> fraction between 0 and 1 used to calcualted whether the change in propensity function
# is acceptable according to the leap condition
# In Gillespies paper this is set to 0.03 
# Is there a way to calculate this automatically?! 
epsi = 0.03
popul_num_all = [popul_num]
propensity = np.zeros(len(LHS))
rxn_vector = np.zeros(len(LHS))  


while tao < tmax:
    propensity = propensity_calc(LHS, popul_num, stoch_rate)        
    a0 = (sum(propensity))
    if a0 == 0.0:
        break
    # if reaction cannot fire corresponding element in rxn_vector should be zero --> tau leaping method 
    lam = (propensity_calc(LHS, popul_num, stoch_rate)*delta_t)
    rxn_vector = np.random.poisson(lam)    
    if tao + delta_t > tmax:
        break
    tao += delta_t
    # divide tao by delta_t to calculate number of leaps
    leap_counter = tao / delta_t  
    # tao leaping part of algoritm --> should be in if statement
    # else clause is the exact SSA! 
    if tao >= 2/a0:     # potential error evaluating elements of an array
        for j in range(len(rxn_vector)):
            state_change_lambda = np.squeeze(np.asarray(state_change_array[j])*rxn_vector[j]) 
            popul_num = popul_num + state_change_lambda
        new_propensity = propensity_calc(LHS, popul_num, stoch_rate)    
        for m in range(len(propensity)):
            for n in range(len(new_propensity)):
                propensity_check = propensity + state_change_lambda 
                if propensity_check - new_propensity >= epsi*a0:  
                    print("The value of delta_t {} choosen is too large".format(delta_t))
                    break
                    else:
                    # check if any num in popul_num is negative must stop simulation and reject leap
                    if popul_num.any() < 0:
                        break   
                    popul_num_all.append(popul_num)     




# use functions to tidey-up code? 

print("Number of leaps, tao, in simulation:\n", leap_counter) 

# Include conditional to use the exact SSA if the value of tao is small enough! 


popul_num_all = np.array(popul_num_all)
for i, label in enumerate(['S1']):
    plt.plot(popul_num_all, label=label)
plt.legend()
plt.tight_layout()
plt.show()