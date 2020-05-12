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
state_change_matrix = RHS - LHS

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

# function to sample the number of reactions that will fire
# from a poisson random variable with parameters(propensity and delta_t) 

# The defined time increment 
# for poisson probability to work must be an integer --> why? 
delta_t = 1

lam = (propensity_calc(LHS, popul_num, stoch_rate)*delta_t)

def t_discretisation_var(lam):
    """ delta_t is lam parameter for np.random.poisson describes 
    the time interval for which the reactions may or may not fire in 
    according to the propensity functions
    """
    #propensity = propensity_calc(LHS, popul_num, stoch_rate)
    # ^^^error here --> propensity is an array!^^^
    #lam = (propensity_calc(LHS, popul_num, stoch_rate)*delta_t) # numpy.ndarray object is not callable
    # lam is numpy.ndarray type
    poisson_distribution = np.random.poisson(lam)
    #print(poisson_distribution)
    return poisson_distribution

 
print("time discretisation_var:\n", t_discretisation_var(lam))
# produces an array of the length 3 for propensity functions of each reaction 
# three reactions == 3 propensity functions
# need to control which value of the array is used in the parameter lam? 


popul_num_all = [popul_num]
propensity = np.zeros(len(LHS))
while tao < tmax:
    propensity = propensity_calc(LHS, popul_num, stoch_rate)        # is this still necessary
    a0 = (sum(propensity))
    if a0 == 0.0:
        break
    delta_t_rxn_prob = t_discretisation_var(lam) # probability of event happening in time period t   
    # cannot call numpy.ndarray object lam
    # tried to call a numpy array as a function
    num_rxn = np.arange(delta_t_rxn_prob.size)
    if tao + delta_t > tmax:
        break
    j = stats.rv_discrete(values=(num_rxn, delta_t_rxn_prob)).rvs()
    #print(tao, t)
    t = t + delta_t
    popul_num = popul_num + np.squeeze(np.asarray(state_change_matrix[j]))
    popul_num_all.append(popul_num)


popul_num_all = np.array(popul_num_all)
for i, label in enumerate(['Enzyme', 'Substrate', 'Enzyme-Substrate complex', 'Product']):
    plt.plot(popul_num_all[:, i], label=label)
plt.legend()
plt.tight_layout()
plt.show()