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

# System of equations
"""
E + S --> ES == 1E + 1S + 0ES + 0P --> 0E + 0S + 1ES + 0P
ES --> E + S == 0E + 0S + 1ES + 0P --> 1E + 1S + 0ES + 0P
ES --> E + P == 0E + 0S + 1ES + 0P --> 1E + 0S + 0ES + 1P
"""
# Need to initialise the discrete numbers of molecules???
popul_num = np.array([200, 100, 0, 0])

# ratios of starting materials for each reaction 
LHS = np.array([[1,1,0,0], [0,0,1,0], [0,0,1,0]])

# ratios of products for each reaction
RHS = np.matrix([[0,0,1,0], [1,1,0,0], [1,0,0,1]])

# stochastic rates of reaction
stoch_rate = np.array([0.0016, 0.0001, 0.1])

# Define the state change vector
state_change_array = np.asarray(RHS - LHS)

# Intitalise time variables
tmax = 100.0         # Maximum time
tao = 0.0           # array to store the time of the reaction.


# function to calcualte the propensity functions for each reaction
def propensity_calc(LHS, popul_num, stoch_rate):
    propensity = np.zeros(len(LHS))
    for row in range(len(LHS)):
            a = stoch_rate[row]     
            for i in range(len(popul_num)):
                if (popul_num[i] >= LHS[row, i]):       
                    binom_rxn = binom(popul_num[i], LHS[row, i])
                    a = a*binom_rxn
                else:
                    a = 0
                    break
            propensity[row] = a     
    return propensity

print(propensity_calc(LHS, popul_num, stoch_rate))



# Tau-leaping  method

delta_t = 0.1
popul_num_all = [popul_num]
propensity = np.zeros(len(LHS))
rxn_vector = np.zeros(len(LHS))  


# need to implement a post leap check on the value of tau! 
# compute the new state 
# calculate the propensity in both old and new state 
# check that the difference is small enough 

# Need some way of storing the old and new propensities
# calculate propensity at start of while loop
# run while loop
# recalculate propensity 
# compare propensity 
# use predefined threshhold to check that the propensities havent changed signigficantly. 


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
    for j in range(len(rxn_vector)):
        state_change_lambda = np.squeeze(np.asarray(state_change_array[j])*rxn_vector[j]) 
        popul_num = popul_num + state_change_lambda
    new_propensity = propensity_calc(LHS, popul_num, stoch_rate)    
    print("propensity:\n", propensity)
    print("new propensity:\n", new_propensity)
    # Is the new propensity calculated properly? 
    # post leap check
    # Goes into if statementat the very last iteration!  
    for m in range(len(propensity)):
        for n in range(len(new_propensity)):
            propensity_check = propensity[m] + state_change_lambda
            #print("propensity check:\n", propensity_check)
            if propensity_check[m] - new_propensity[n] >= a0:  # should i specify an element in a0 to check against?
                # need to check conditional logic! 
                print("The value of delta_t {} choosen is too large".format(delta_t))
                break
            else:
                # check if any num in popul_num is negative must stop simulation and reject leap
                if popul_num.any() < 0:
                    break   
                popul_num_all.append(popul_num)    

# Doesn't go into if statement! 


#print("propensity[m]:\n", propensity[m])
#print("propensity[n]:\n", new_propensity[n])


popul_num_all = np.array(popul_num_all)
for i, label in enumerate(['Enzyme', 'Substrate', 'Enzyme-Substrate complex', 'Product']):
    plt.plot(popul_num_all[:, i], label=label)
plt.legend()
plt.tight_layout()
plt.show()

# Need to lable axis 