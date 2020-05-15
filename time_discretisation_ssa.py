# https://github.com/karinsasaki/gillespie-algorithm-python/blob/master/build_your_own_gillespie_exercises.ipynb
# ^^^ Github example of stochastic implementation ^^^
# http://be150.caltech.edu/2016/handouts/gillespie_simulation.html
# ^^^ Example algorithm ^^^

# Implement a time discretisation variant of the SSA 
# Using the poisson distribution to select  

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
print(state_change_array)

# Intitalise time variables
tmax = 20.0         # Maximum time
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



# outcome of algorithm step 2: 
# reaction vector, r --> The number of reactions in the model
#   simulate the reaction vector with the ith entry of the poisson distribution
# A state change vector A = R - L <-- already in code
delta_t = 1.3
popul_num_all = [popul_num]
propensity = np.zeros(len(LHS))


while tao < tmax:
    propensity = propensity_calc(LHS, popul_num, stoch_rate)        
    a0 = (sum(propensity))
    if a0 == 0.0:
        break
    lam = (propensity_calc(LHS, popul_num, stoch_rate)*delta_t)
    rxn_vector = np.random.poisson(lam)  # number of reactions that can fire in time step delta_t
    #print("reaction vector:\n", rxn_vector)     
    if tao + delta_t > tmax:
        break
    tao += delta_t
    for row in range(len(state_change_array)):
        for j in range(len(rxn_vector)):
            popul_num = popul_num + np.squeeze(np.asarray(row*j)) 
    print(popul_num, tao)
    # population numbers of Enzyme and substrate are going up! 
    # ^^^ That is wrong^^^
    popul_num_all.append(popul_num) 

# what happens to the Enzyme-substrate complex? Doesn't print 
# Is multiplication right?  

# creates straight line graphs that only increase! 
# wrong! 


popul_num_all = np.array(popul_num_all)
for i, label in enumerate(['Enzyme', 'Substrate', 'Enzyme-Substrate complex', 'Product']):
    plt.plot(popul_num_all[:, i], label=label)
plt.legend()
plt.tight_layout()
plt.show()