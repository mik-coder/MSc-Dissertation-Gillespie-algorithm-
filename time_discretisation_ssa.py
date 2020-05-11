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


# define a fixed time interval at the outset
# for each reaction in LHS sample from the poisson distribution the number of reactions to fire 
# simulate all the reactions in one go
 

# function to calcualte the propensity functions for each reaction
def propensity_calc(LHS, popul_num, stoch_rate):
    propensity = np.zeros(len(LHS))
    for row in range(len(LHS)):
            a = stoch_rate[row]     # type = numpy.float64
            for i in range(len(popul_num)):
                if (popul_num[i] >= LHS[row, i]):       # seems to work --> iterates through multiple elementa
                    # will return a new array/ value with just true or false --> How to use this further
                    binom_rxn = binom(popul_num[i], LHS[row, i])
                    a = a*binom_rxn
                else:
                    a = 0
                    break
            propensity[row] = a     # type = numpy.ndarray
    return propensity



# function to sample the number of reactions that will fire
# from a poisson random variable with parameters(propensity and delta_t) 

# The defined time increment 
# for poisson probability to work must be an integer --> why? 
delta_t = 1.2

def t_discretisation_var():
    # sample number of rxns to fire in given time period using poisson distribution 
    # delta_t is the expectation interval
    poisson_prob = np.random.poisson(propensity_calc(LHS, popul_num, stoch_rate), delta_t)
    print(poisson_prob)
    print(type(poisson_prob))
    return poisson_prob


t_discretisation_var()
# returns one integer value, not sure what it is?  
# poisson prob is a numpy.ndarray 


# simulate all reactions at once --> change while loop? 
# while simulates one reaction after the other using iteration 


popul_num_all = [popul_num]
propensity = np.zeros(len(LHS))
while tao < tmax:
    propensity = propensity_calc(LHS, popul_num, stoch_rate)
    a0 = (sum(propensity))
    if a0 == 0.0:
        break
    t = np.random.exponential(1/a0) # sample time from random exponential --> now needs to change to delta_t
    rxn_probability = propensity / a0   # propensity = array a0 = number --> Error
    num_rxn = np.arange(rxn_probability.size)
    if tao + t > tmax:
        tao = tmax
        break
    j = stats.rv_discrete(values=(num_rxn, rxn_probability)).rvs()
    #print(tao, t)
    tao = tao + t
    popul_num = popul_num + np.squeeze(np.asarray(state_change_matrix[j]))
    popul_num_all.append(popul_num)


popul_num_all = np.array(popul_num_all)
for i, label in enumerate(['Enzyme', 'Substrate', 'Enzyme-Substrate complex', 'Product']):
    plt.plot(popul_num_all[:, i], label=label)
plt.legend()
plt.tight_layout()
#plt.show()