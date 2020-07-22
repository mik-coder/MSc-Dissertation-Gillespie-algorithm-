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
tmax = 15.0         # Maximum time --> change to match x-axis in paper         

def propensity_calc(LHS, popul_num, stoch_rate):
    propensity = np.zeros(len(LHS))
    for row in range(len(LHS)):
            a = stoch_rate[row]     # type = numpy.float64
            for i in range(len(popul_num)):
                if (popul_num[i] >= LHS[row]):      
                    binom_rxn = binom(popul_num[i], LHS[row])
                    a = a*binom_rxn
                else:
                    a = 0
                    break
            propensity[row] = a     # type = numpy.ndarray
    return propensity




def gillespie_exact_ssa(initial_state, LHS, stoch_rate, state_change_array):
    popul_num = initial_state
    popul_num_all = [initial_state]
    propensity = np.zeros(len(LHS))
    tao = 0.0 
    tao_all = [tao] 
    while tao < tmax:
        propensity = propensity_calc(LHS, popul_num, stoch_rate)
        a0 = (sum(propensity))
        if a0 == 0.0:
            break
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
    print("tao:\n", tao)
    print("Molecule numbers:\n", popul_num)
    popul_num_all = np.array(popul_num_all)
    tao_all = np.array(tao_all)
    return popul_num_all, tao_all

def parallelfunc(v):
    gillespie_exact_ssa(start_state, LHS, stoch_rate, state_change_array)

if __name__ == '__main__':
    start = datetime.utcnow()
    with Pool(processes= 2) as p:
        pool_results = p.map(parallelfunc, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50])
    end = datetime.utcnow()
    sim_time = end - start
    print(f"Simualtion utc time:\n{sim_time}")



print("Parallel Simulation\nSystem 1")

def gillespie_plot(start_state, LHS, stoch_rate, state_change_array):
    """ Function to plot the results of the Gillespie simulation"""
    fig, ax = plt.subplots()
    popul_num_all, tao_all = gillespie_exact_ssa(start_state, LHS, stoch_rate, state_change_array)
    for i, label in enumerate(['S1']):
        ax.plot(tao_all, popul_num_all, label=label)
    plt.legend()
    plt.show()
    return fig


gillespie_plot(start_state, LHS, stoch_rate, state_change_array)