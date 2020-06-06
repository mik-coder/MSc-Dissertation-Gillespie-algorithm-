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

# tau-leaping SSA variant of an Irriversible Isomerisam process!  
# Need to make custom class to time algorithm! 
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
        # should stop the timer!
        elasped_simulation_time = self._simulation_stop_time - self._simulation_start_time  
        self._simulation_start_time = None
        print(f"Elasped time: {elasped_simulation_time:0.4f} seconds")

# Must make sure that all instantiated variables are not already builtin functions

# System of equations
""" S1 --> 0 == 1 --> 0
"""

# Need to initialise the discrete numbers of molecules???
popul_num = np.array([1.0E5])
# should use --> [1.0E5]

# ratios of starting materials for each reaction 
LHS = np.array([1])

# ratios of products for each reaction
RHS = np.array([0])

# stochastic rates of reaction
stoch_rate = np.array([1.0])

# Define the state change vector
state_change_array = np.asarray(RHS - LHS)

# Intitalise time variables
tmax = 5.0         # Maximum time
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

# Tau-leaping  method
epsi = 0.03
popul_num_all = [popul_num]
tao_all = [tao]
propensity = np.zeros(len(LHS))
rxn_vector = np.zeros(len(LHS))  


# Need an ad-hoc model specific determination of delta_t 
"""from the binomial formula: [x(x - 1) / 1!] + 1 where x is the number of molecules 
This is exanded to: x^2 - x + 1 
This is then differentiatied to: 2x - 1 
The molecule numbers are substituted in for x to give: 199999 """

# define a function to calcualte the value best value of delta_t
def time_step_count(propensity_calc, state_change_array, epsi): 
    evaluate_propensity = propensity_calc(LHS, popul_num, stoch_rate) 
    a0 = sum(evaluate_propensity)
    # equation 22:   
    exptd_state_change = sum(evaluate_propensity*state_change_array)  
    # TypeError: numpy.float64 is not iterable
    # Both are numpy.ndarray --> why the error?
    # equation 24: 
    bj = evaluate_propensity / 199999
    # equation 26: 
    delta_t = ((epsi*a0)/abs(exptd_state_change*bj))
    print("The calculated value of delta_t:\n", delta_t)
    return delta_t



delta_t = time_step_count(propensity_calc, state_change_array, epsi)


def gillespie_tau_leaping(propensity_calc, popul_num, popul_num_all, tao_all, rxn_vector, tao, delta_t, epsi):
    t = simulation_timer()
    t.start()
    while tao < tmax:
        propensity = propensity_calc(LHS, popul_num, stoch_rate)        
        a0 = (sum(propensity))
        if a0 == 0.0:
            break
        # if reaction cannot fire corresponding element in rxn_vector should be zero --> tau leaping method 
        if popul_num.any() < 0:
            print("Number of molecules {} is too small for reaction to fire".format(popul_num))
            break   
        lam = (propensity_calc(LHS, popul_num, stoch_rate)*delta_t)
        rxn_vector = np.random.poisson(lam)    
        if tao + delta_t > tmax:
            break
        tao += delta_t
        # divide tao by delta_t to calculate number of leaps  
        if tao >= 2/a0:     
            for j in range(len(rxn_vector)):
                state_change_lambda = np.squeeze(np.asarray(state_change_array[j])*rxn_vector[j]) 
                popul_num = popul_num + state_change_lambda
                new_propensity = propensity_calc(LHS, popul_num, stoch_rate)   
            for n in range(len(new_propensity)):
                propensity_check = propensity + state_change_lambda 
                if propensity_check[n] - new_propensity[n] >= epsi*a0:  
                    print("The value of delta_t {} choosen is too large".format(delta_t))
                    break
                else:
                    popul_num = popul_num + state_change_lambda     
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
    print("tao:\n", tao)
    print("Molecule numbers:\n", popul_num)
    leap_counter = tao / delta_t 
    print("Number of leaps taken:\n", leap_counter)
    t.stop()
    return popul_num_all.append(popul_num), tao_all.append(tao)


# RUNS!!!
# leap_output not quite right 
 
print(gillespie_tau_leaping(propensity_calc, popul_num, popul_num_all, tao_all, rxn_vector, tao, delta_t, epsi))

popul_num_all = np.array(popul_num_all)
tao_all = np.array(tao_all)
for i, label in enumerate(['S1']):
    plt.plot(popul_num_all, label=label)
plt.legend()
plt.tight_layout()
plt.show()

# Updated version!
# Needs pushing up to Github