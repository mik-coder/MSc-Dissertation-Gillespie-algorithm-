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

# tau-leaping SSA variant  

# System of equations
# stochiometric equation needs to have equal length! 
""" 1S + 0T + 0U --> 0S + 0T + 0U
    2S + 0T + 0U --> 0S + 1T + 0U 
    0S + 1T + 0U --> 2S + 0T + 0U
    0S + 1T + 0U --> 0S + 0T + 1U   
"""

# Need to initialise the discrete numbers of molecules???
popul_num = np.array([1.0E5, 0, 0])
# adding in another 1.0E5 --> Gives index error in 52 --> if condition of propensity_calc function   

# ratios of starting materials for each reaction 
LHS = np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 1, 0]])

# ratios of products for each reaction
RHS = np.array([[0, 0, 0], [0, 1, 0], [2, 0, 0], [0, 0, 1]])

# stochastic rates of reaction
stoch_rate = np.array([1.0, 0.002, 0.5, 0.04])

# Define the state change vector
state_change_array = np.asarray(RHS - LHS)  
print("State change array:\n", state_change_array)   
# state change array has length 4 
# 

# Intitalise time variables
tmax = 30.0         # Maximum time
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

print("propensity_calc:\n", (propensity_calc(LHS, popul_num, stoch_rate).shape))



# Tau-leaping  method

delta_t = 0.1
epsi = 0.03
popul_num_all = [popul_num]
propensity = np.zeros(len(LHS))
rxn_vector = np.zeros(len(LHS))  


while tao < tmax:
    propensity = propensity_calc(LHS, popul_num, stoch_rate)        
    a0 = (sum(propensity))  # a0 is a numpy.float64
    if a0 == 0.0:
        break   # if reaction cannot fire corresponding element in rxn_vector should be zero --> tau leaping method 
    lam = (propensity_calc(LHS, popul_num, stoch_rate)*delta_t)  
    rxn_vector = np.random.poisson(lam)
    if tao + delta_t > tmax:
        break
    tao += delta_t  # divide tao by delta_t to calculate number of leaps
    leap_counter = tao / delta_t 
    if popul_num.any() < 0:
        break
    else:   
        if tao >= 2/a0:     
            for j in range(len(rxn_vector)): 
                state_change_lambda = np.squeeze(np.asarray(state_change_array[j])*rxn_vector[j]) 
            new_propensity = propensity_calc(LHS, popul_num, stoch_rate)  
            propensity_check = propensity.copy()
            propensity_check[0] += state_change_lambda[0]
            propensity_check[1:] += state_change_lambda 
            leap_check = propensity_check - new_propensity
            for n in range(len(leap_check)):
                if leap_check[n] >= epsi*a0:  
                    print("The value of delta_t {} choosen is too large".format(delta_t))
                    break  
            popul_num = popul_num + state_change_lambda                  
            popul_num_all.append(popul_num)   
        else:
            t = np.random.exponential(1/a0)
            rxn_probability = propensity / a0   
            num_rxn = np.arange(rxn_probability.size)       
            if tao + t > tmax:      
                tao = tmax
                break
            j = stats.rv_discrete(values=(num_rxn, rxn_probability)).rvs() 
            tao = tao + t
            popul_num = popul_num + np.squeeze(np.asarray(state_change_array[j]))  
            print("Simulation time:\n", t, tao)
            popul_num_all.append(popul_num) 


# Runs but the plotting isnt right! 
# Only plots S and U as constant values 
# As if no reaction is happening!
# problem sampling reactions and updating the system! 

# first check indentation --> and updating popul_num_all!


print("Number of leaps, tao, in simulation:\n", leap_counter) 


popul_num_all = np.array(popul_num_all)
for i, label in enumerate(['S', 'T', 'U']):
    plt.plot(popul_num_all[:, i], label=label)
plt.legend()
plt.tight_layout()
plt.show()


# JOB FOR ANOTHER DAY! 
# Each SSA variant should be a seperate function
# avoid using excess loops and makes it easier to debug! 
