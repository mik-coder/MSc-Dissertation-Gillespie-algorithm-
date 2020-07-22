import numpy as np
import scipy
from scipy.special import binom
from scipy import stats
from matplotlib import pyplot as plt # allow inline plot with %matplotlib inline
import time
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Pool

# Exact SSA of an Irriversible Isomerisam process!  

# System of equations
""" S1 --> 0 == 1 --> 0
"""

# custom timer class

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


# Need to initialise the discrete numbers of molecules???
start_state = np.array([1.0E5])
# should use --> [1.0E5]

# ratios of starting materials for each reaction 
LHS = np.array([1])

# ratios of products for each reaction
RHS = np.array([0])

# stochastic rates of reaction
stoch_rate = np.array([1.0])

# Define the state change vector
state_change_array = RHS - LHS      # should be -1

# Intitalise time variables
tmax = 30.0         # Maximum time --> change to match x-axis in paper         


#function to calcualte the propensity functions for each reaction
def propensity_calc(LHS, popul_num, stoch_rate):
    """ Function to calculate to probability of a reaction 
    occuring given its rate and the current molecule numbers"""
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

# Time discretisation method
def t_discreteisation(initial_state, LHS, stoch_rate, state_change_array):
    """Function to simulate the time discretisation variant of the SSA """
    tao = 0.0
    delta_t = 0.01
    tao_all = [tao]
    popul_num = initial_state
    popul_num_all = [initial_state]
    propensity = np.zeros(len(LHS))
    leap_counter = 0
    while tao < tmax:
        propensity = propensity_calc(LHS, popul_num, stoch_rate)        
        a0 = (sum(propensity))
        if a0 == 0.0:
            break
        if (popul_num < 0).any():
            break       
        lam = (propensity_calc(LHS, popul_num, stoch_rate)*delta_t)
        rxn_vector = np.random.poisson(lam)    
        if tao + delta_t > tmax:
            break
        else:
            tao += delta_t  
            for j in range(len(rxn_vector)):
                popul_num = popul_num + np.squeeze(np.asarray(state_change_array[j])*rxn_vector[j]) 
            tao_all.append(tao)
            popul_num_all.append(popul_num) 
            leap_counter += 1
    popul_num_all = np.array(popul_num_all)
    tao_all = np.array(tao_all)
    print("Time of final reaction:\n", tao)
    print("Molecule numbers:\n", popul_num)
    print("Number of leaps:\n", leap_counter)
    return popul_num_all, tao_all

def parallel_func(v):
    t_discreteisation(start_state, LHS, stoch_rate, state_change_array)

if __name__ == '__main__':
    start = time.time()
    with Pool() as p:  
        pool_results = p.map(parallel_func, [1, 2, 3, 4, 5])
    end = time.time()
    sim_time = end - start
    print("Simulation time:\n", sim_time)



def gillespie_plot(start_state, LHS, stoch_rate, state_change_array):
    """ Function to plot the results of the Gillespie simulation"""
    fig, ax = plt.subplots()
    popul_num_all, tao_all = t_discreteisation(start_state, LHS, stoch_rate, state_change_array)
    for i, label in enumerate(['S1']):
        ax.plot(tao_all, popul_num_all, label=label)
    plt.legend()
    plt.show()
    return fig


gillespie_plot(start_state, LHS, stoch_rate, state_change_array)