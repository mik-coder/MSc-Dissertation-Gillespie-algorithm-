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
    # ^^^ rxn_vector length = 4    
    print("rxn_vector length:\n", rxn_vector)
    if tao + delta_t > tmax:
        break
    tao += delta_t  # divide tao by delta_t to calculate number of leaps
    leap_counter = tao / delta_t 
    if popul_num.any() < 0:
        break
    else:   
        if tao >= 2/a0:     
            for j in range(len(rxn_vector)): 
                state_change_lambda = (np.asarray(state_change_array[j])*rxn_vector[j]) # shape (3,) and length = 3 
                # IndexError: index 3 is out of bounds for axis 1 with length 3
                print("state change lambda:\n", len(state_change_lambda))
                # state change lambda --> change in the number of molecules after each reaction is fired --> len = 3 
                # state change lambda needs to be length four! This is where my problem is! 
                # rxn_vector is length 4!  
                popul_num = popul_num + state_change_lambda
            new_propensity = propensity_calc(LHS, popul_num, stoch_rate) # shape = (4,)   
            for m in range(len(propensity)):
                for n in range(len(new_propensity)):
                    # remove one of the nested for loops --> which one? 
                    # both propensity and new_propensity have len = 4 
                    propensity_check = propensity[m] + state_change_lambda # shape = (3,) len = 3
                    print("propensity_check:\n", len(propensity_check))
                    # propensity is len/shape = 4/(4,) 
                    # state change lambda and therefore propensity_check are len/shape = 3/(3,)
                    # the longer is fitted to the shorter for propensity check --> this is the Error
                    # state_change_lambda is the problem --> shorter length --> ONLY THREE MOLECULE TYPES IN SYSTEM!
                    if propensity_check[m] - new_propensity[n] >= epsi*a0:  # IndexError: index 3 is out of bounds for axis zero with size 3
                        print("The value of delta_t {} choosen is too large".format(delta_t))
                        break                    
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


# Dont know if it's going into the if statement or not??? 
# How can I check it works properly? 


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
