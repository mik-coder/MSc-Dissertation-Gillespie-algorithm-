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
import time

# System of equations
"""
E + S --> ES == 1E + 1S + 0ES + 0P --> 0E + 0S + 1ES + 0P
ES --> E + S == 0E + 0S + 1ES + 0P --> 1E + 1S + 0ES + 0P
ES --> E + P == 0E + 0S + 1ES + 0P --> 1E + 0S + 0ES + 1P
"""

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
        elapsed_simulation_time = self._simulation_stop_time - self._simulation_start_time  
        self._simulation_start_time = None
        print(f"Elapsed time: {elapsed_simulation_time:0.4f} seconds")


# Need to initialise the discrete numbers of molecules???
popul_num = np.array([200, 100, 0, 0])

# ratios of starting materials for each reaction 
LHS = np.array([[1,1,0,0], [0,0,1,0], [0,0,1,0]])

# ratios of products for each reaction
RHS = np.array([[0,0,1,0], [1,1,0,0], [1,0,0,1]])

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


def update_array(popul_num, stoch_rate): 
    """Specific to this model 
    will need to change if different model 
    implements equaiton 24 of the Gillespie paper"""
    b = np.array([[stoch_rate[0]*popul_num[0], stoch_rate[0]*popul_num[1], 0.0, 0.0], [0.0, 0.0, 0.0001, 0.0], [0.0, 0.0, 0.1, 0.0]])
    return b
b = update_array(popul_num, stoch_rate)

epsi = 0.03

# returns the value to increment time by --> delta_t
def time_step_calc(propensity_calc, state_change_array, b, epsi):
    """ Function to calculate the simulation 
    time increment delta_t"""
    propensity = propensity_calc(LHS, popul_num, stoch_rate) 
    print("time_step_calc propensity:\n", propensity)
    denominator = np.zeros(len(propensity))
    a0 = sum(propensity)
    # equation 22:
    exptd_state_array = 0.0
    for x in range(len(propensity)):
        exptd_state_array +=  propensity[x]*state_change_array[x]
        # expected states for first two reactions are very large
    print("expectd state array:\n", exptd_state_array) 
    # equation 24: Calculated ad-hoc results in bji matrix
    numerator = epsi*a0 
    for j in range(len(propensity)):
        for i in range(len(popul_num)):
            denominator[j] += (exptd_state_array[i]*b[j, i])
            # ValueError: setting an array element with a sequence
            # Type Error only size-1 arrays can be converted to python scalars --> direct cause of the above error
    # equation 26
    delta_t_array = numerator/abs(denominator)
    print("delta_t_array:\n", delta_t_array)
    delta_t = min(delta_t_array)
    print("The calculated value of delta_t:\n", delta_t)
    return delta_t
delta_t = time_step_calc(propensity_calc, state_change_array, b, epsi)



# Tau-leaping  method
# Need to check the main propensity calculation 
# Need to put into a function! 
popul_num_all = [popul_num]
tao_all = [tao]
propensity = np.zeros(len(LHS))
rxn_vector = np.zeros(len(LHS))  


def gillespie_tau_leaping(propensity_calc, popul_num, LHS, stoch_rate, popul_num_all, tao_all, rxn_vector, delta_t, tao, epsi): 
    t = simulation_timer()
    t.start()
    leap_counter = 0
    while tao < tmax:
        propensity = propensity_calc(LHS, popul_num, stoch_rate)        
        a0 = (sum(propensity))
        if a0 == 0.0:
            break
        if popul_num.any() < 0:
            break   
        lam = (propensity_calc(LHS, popul_num, stoch_rate)*delta_t)
        rxn_vector = np.random.poisson(lam)   
        if tao + delta_t > tmax:
            break
        tao += delta_t  
        if delta_t >= 1/a0:
            for j in range(len(rxn_vector)):
                state_change_lambda = np.squeeze(np.asarray(state_change_array[j])*rxn_vector[j]) 
                popul_num = popul_num + state_change_lambda
            leap_counter += 1 
            popul_num_all.append(popul_num)
            tao_all.append(tao)     
        else: 
            next_t = np.random.exponential(1/a0)
            rxn_probability = propensity / a0   
            num_rxn = np.arange(rxn_probability.size)       
            if tao + next_t > tmax:      
                tao = tmax
                break
            j = stats.rv_discrete(values=(num_rxn, rxn_probability)).rvs() 
            tao = tao + next_t
            popul_num = popul_num + np.squeeze(np.asarray(state_change_array[j]))  
            popul_num_all.append(popul_num)   
            tao_all.append(tao)  
    print("Molecule numbers:\n", popul_num)
    print("Time of final simulation:\n", tao)
    print("leap counter:\n", leap_counter)
    t.stop()
    return popul_num_all.append(popul_num), tao_all.append(tao)


print(gillespie_tau_leaping(propensity_calc, popul_num, LHS, stoch_rate, popul_num_all, tao_all, rxn_vector, delta_t, tao, epsi))
 


popul_num_all = np.array(popul_num_all)
tao_all = np.array(tao_all)

for i, label in enumerate(['Enzyme', 'Substrate', 'Enzyme-Substrate complex', 'Product']):
    plt.plot(tao_all, popul_num_all[:, i], label=label)
plt.legend()
plt.tight_layout()
plt.show()

