import numpy as np
import scipy
from scipy.special import binom
from scipy import stats
from matplotlib import pyplot as plt # allow inline plot with %matplotlib inline
import time
import multiprocessing as mp
from multiprocessing import Pool
from datetime import datetime

# Exact SSA of an Irriversible Isomerisam process!  

# System of equations
# stochiometric equation needs to have equal length! 
""" 1S + 0T + 0U --> 0S + 0T + 0U
    2S + 0T + 0U --> 0S + 1T + 0U 
    0S + 1T + 0U --> 2S + 0T + 0U
    0S + 1T + 0U --> 0S + 0T + 1U   
"""

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
        print(f"Elasped time: {elasped_simulation_time:0.4f} seconds")


# Need to initialise the discrete numbers of molecules???
start_state = np.array([1.0E5, 0, 0])

# ratios of starting materials for each reaction 
LHS = np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 1, 0]])

# ratios of products for each reaction
RHS = np.array([[0, 0, 0], [0, 1, 0], [2, 0, 0], [0, 0, 1]])

# stochastic rates of reaction
stoch_rate = np.array([1.0, 0.002, 0.5, 0.04])

# Define the state change vector
state_change_array = np.asarray(RHS - LHS)     

# Intitalise time variables
tmax = 50.0         # Maximum time         

def propensity_calc(LHS, popul_num, stoch_rate):
    propensity = np.zeros(len(LHS))
    for row in range(len(LHS)):
            a = stoch_rate[row]     # type = numpy.float64
            for i in range(len(popul_num)):
                if (popul_num[i] >= LHS[row, i]):      
                    binom_rxn = binom(popul_num[i], LHS[row, i])
                    a = a*binom_rxn
                else:
                    a = 0
                    break
            propensity[row] = a     
    return propensity


def gillespie_exact_ssa(initial_state, LHS, stoch_rate, state_change_array):
    popul_num_all = [initial_state]
    propensity = np.zeros(len(LHS))
    tao = 0.0  
    tao_all = [tao]
    popul_num = initial_state
    while tao < tmax:
        propensity = propensity_calc(LHS, popul_num, stoch_rate)
        a0 = (sum(propensity))
        if a0 == 0.0:
            break
        if popul_num.any() < 0:
            print("Number of molecules {} is too small for reaction to fire".format(popul_num))
            break 
        next_t = np.random.exponential(1/a0)     # sampling time of next reaction here --> time smaller ever time when rate is bigger! 
        rxn_probability = propensity / a0   
        num_rxn = np.arange(rxn_probability.size)       
        if tao + next_t > tmax:
            break
        j = stats.rv_discrete(values=(num_rxn, rxn_probability)).rvs()      # sampling next reaction to occur here
        tao = tao + next_t
        popul_num = popul_num + np.squeeze(np.asarray(state_change_array[j]))   # add state change array for that reaction!
        popul_num_all.append(popul_num) 
        tao_all.append(tao) 
    print("tao:\n", tao)
    print("Molecule numbers:\n", popul_num)
    print("Number of reactions:\n", len(tao_all))
    popul_num_all = np.array(popul_num_all)
    tao_all = np.array(tao_all)
    return popul_num_all, tao_all


def parallel_func(v):
    gillespie_exact_ssa(start_state, LHS, stoch_rate, state_change_array)

if __name__ == '__main__':
    start = datetime.utcnow()
    with Pool() as p:
        pool_results = p.map(parallel_func, [1, 2, 3, 4, 5])
    end = datetime.utcnow()
    sim_time = end - start
    print("Simulation time:\n", sim_time)


def gillespie_plot(start_state, LHS, stoch_rate, state_change_array):
    """Function to plot hte results of the gillespie simualtion""" 
    popul_num_all, tao_all = gillespie_exact_ssa(start_state, LHS, stoch_rate, state_change_array)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(tao_all, popul_num_all[:, 0], label='S', color= 'Green')
    ax1.legend()
    for i, label in enumerate(['T', 'U']):
        ax2.plot(tao_all, popul_num_all[:, i+1], label=label)
    ax2.legend()
    plt.tight_layout()
    plt.show()
    return fig


gillespie_plot(start_state, LHS, stoch_rate, state_change_array)

