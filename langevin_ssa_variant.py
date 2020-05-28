# Implement a tau-leaping variant of the SSA   

import numpy as np
import scipy
from scipy.special import binom
from scipy import stats
from matplotlib import pyplot as plt # allow inline plot with %matplotlib inline
import time

# tau-leaping SSA variant 
# Class for starting the timer

class TimeError(Exception):
    """A custom exception used to report errors in use of Timer Class""" 

class simulation_timer: 
    def __init__(self):
        self._simulation_start_time = None
        self._simulation_stop_time = None

    def start(self):
        """start a new timer"""
        if self._simulation_start_time is not None:    # attribute
            raise TimeError(f"Timer is running.\n Use .stop() to stop it")

        self._simulation_start_time = time.perf_counter()  
    def stop(self):
        """stop the time and report the elsaped time"""
        if self._simulation_start_time is None:
            raise TimeError(f"Timer is not running.\n Use .start() to start it.")

        self._simulation_stop_time = time.perf_counter()
        elasped_simulation_time = self._simulation_stop_time - self._simulation_start_time  
        self._simulation_start_time = None
        print(f"Elasped time: {elasped_simulation_time:0.8f} seconds")

# System of equations
# stochiometric equation needs to have equal length! 
""" 1S + 0T + 0U --> 0S + 0T + 0U
    2S + 0T + 0U --> 0S + 1T + 0U 
    0S + 1T + 0U --> 2S + 0T + 0U
    0S + 1T + 0U --> 0S + 0T + 1U   
"""

# Need to implement the langevin variant method! 


# Need to initialise the discrete numbers of molecules
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

# Tau-leaping while-loop method

delta_t = 1.2E-4
epsi = 0.03
popul_num_all = [popul_num]
tao_all = [tao]
propensity = np.zeros(len(LHS))
rxn_vector = np.zeros(len(LHS))  

# runs but the plot isn't good enough! 
# look for paramerters in model and rest of method 

def gillespie_tau_leaping(propensity_calc, popul_num, popul_num_all, tao_all, rxn_vector, tao, delta_t, epsi):
    t = simulation_timer()
    t.start()
    while tao < tmax:
        propensity = propensity_calc(LHS, popul_num, stoch_rate)        
        a0 = (sum(propensity))  # a0 is a numpy.float64
        if a0 == 0.0: 
            break   
        if popul_num.any() < 0: # This isnt working
            print("Number of molecules {} is too small for reaction to fire".format(popul_num))
            break
        # for Langevin this needs to change
        normal_ran_var = np.random.normal(0, 1) 
        rxn_vector = ((propensity*tao) + (propensity*tao)**0.5)*normal_ran_var
        # Is the result going to be a vector ? 
        if tao + delta_t > tmax:
            break    
        else:   # otherwise...
            tao += delta_t 
            if tao >= 2/a0:     # if tao is big enough
                for j in range(len(rxn_vector)):  
                    state_change_lambda = np.squeeze(np.asarray(state_change_array[j])*rxn_vector[j]) 
                    # caclulating the state change after a reaction haas fired  
                    new_propensity = propensity_calc(LHS, popul_num, stoch_rate)
                    propensity_check = propensity.copy()
                    propensity_check[0] += state_change_lambda[0]
                    propensity_check[1:] += state_change_lambda  
                    # checking that selected value of tao is not too big! 
                    for n in range(len(propensity_check)): 
                        if propensity_check[n] - new_propensity[n] >= epsi*a0:   
                            print("The value of delta_t {} choosen is too large".format(delta_t))
                            break  
                        else:
                            popul_num = popul_num + state_change_lambda                  
                            popul_num_all.append(popul_num)
                            tao_all.append(tao) 
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
                popul_num_all.append(popul_num)   
                tao_all.append(tao)
        leap_counter = tao / delta_t    # divide tao by delta_t to calculate number of leaps
    print("tao:\n", tao)
    print("Molecule numbers:\n", popul_num)
    t.stop()        # error here when delta_t is 
    return popul_num_all.append(popul_num), tao_all.append(tao), leap_counter


print(gillespie_tau_leaping(propensity_calc, popul_num, popul_num_all, tao_all, rxn_vector, tao, delta_t, epsi))




popul_num_all = np.array(popul_num_all)
tao_all = np.array(tao_all)
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(tao_all, popul_num_all[:, 0], label='S', color= 'Green')
ax1.legend()

for i, label in enumerate(['T', 'U']):
    ax2.plot(tao_all, popul_num_all[:, i+1], label=label)

ax2.legend()
plt.tight_layout()
plt.show()
