# Implement a Langevin variant of the SSA   
import numpy as np
import scipy
from scipy.special import binom
from scipy import stats
from matplotlib import pyplot as plt # allow inline plot with %matplotlib inline
import time
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Pool

# tau-leaping SSA variant 
# Class for starting the timer
class TimeError(Exception):
    """A custom exception used to report errors in use of Timer Class""" 
    pass

class SimulationTimer: 
    accumulated_elapsed_time = 0.0  # Needs to be a class variable not an instance variable
    
    def __init__(self):
        self._simulation_start_time = None
        self._simulation_stop_time = None
        #self.accumulated_elapsed_time = 0.0 #--> Needs to be a CLASS variable

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
        self.accumulated_elapsed_time += elapsed_simulation_time

        self._simulation_start_time = None
        print(f"Elapsed time: {elapsed_simulation_time:0.10f} seconds")
    
    def get_accumulated_time(self):
        """ Return the elapsed time for later use"""
        return self.accumulated_elapsed_time

# System of equations
# stochiometric equation needs to have equal length! 
""" 1S + 0T + 0U --> 0S + 0T + 0U
    2S + 0T + 0U --> 0S + 1T + 0U 
    0S + 1T + 0U --> 2S + 0T + 0U
    0S + 1T + 0U --> 0S + 0T + 1U   
"""

# Need to initialise the discrete numbers of molecules???
start_state = np.array([1.0E5, 0, 0])
# adding in another 1.0E5 --> Gives index error in 52 --> if condition of propensity_calc function   

# ratios of starting materials for each reaction 
LHS = np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 1, 0]])

# ratios of products for each reaction
RHS = np.array([[0, 0, 0], [0, 1, 0], [2, 0, 0], [0, 0, 1]])

# stochastic rates of reaction
stoch_rate = np.array([1.0, 0.002, 0.5, 0.04])

# Define the state change vector
state_change_array = np.asarray(RHS - LHS)   


# Intitalise time variables
tmax = 50.0        # Maximum time


# function to calcualte the propensity functions for each reaction
def propensity_calc(LHS, popul_num, stoch_rate):
    """Function to calculate the probabilty of a reactin in the system firing """
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


def update_array(popul_num, stoch_rate): 
    """Specific to this model 
    will need to change if different model 
    implements equaiton 24 of the Gillespie paper""" 
    s_derviative = stoch_rate[1]*(2*popul_num[0] -1)/2
    b = np.array([[1.0, 0.0, 0.0], [s_derviative, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.4, 0.0]])
    return b 


epsi = 0.03     

def time_step_calc(popul_num, stoch_rate, state_change_array, b, epsi):
    """ Function to calculate the simulation 
    time increment delta_t""" 
    propensity = propensity_calc(LHS, popul_num, stoch_rate)
    denominator = np.zeros(len(propensity))
    a0 = sum(propensity)
    # equation 22: --> propensity*expected state
    exptd_state_array = 0.0     
    for x in range(len(propensity)):    
        exptd_state_array += propensity[x]*state_change_array[x]    
    # equation 24: Calculated ad-hoc results in bji matrix 
    for j in range(len(propensity)):
        for i in range(len(popul_num)):
            denominator[j] += (exptd_state_array[i]*b[j, i])   
            checked_denominator = denominator[denominator != 0]   
    # equation 26
    numerator = epsi*a0
    delta_t_array = (numerator/abs(checked_denominator))    
    delta_t = min(delta_t_array)
    return delta_t


def gillespie_langevin(initial_state, LHS, stoch_rate, state_change_array):
    """ Function to run the Gillespie stochastic simulation Langevin variant """
    popul_num = initial_state
    popul_num_all = [initial_state]
    propensity = np.zeros(len(LHS))
    rxn_vector = np.zeros(len(LHS))
    tao = 0.0
    tao_all = [tao]
    leap_counter = 0
    while tao < tmax:
        propensity = propensity_calc(LHS, popul_num, stoch_rate)      
        a0 = (sum(propensity))
        if a0 == 0.0:
            break
        b = update_array(popul_num, stoch_rate) 
        delta_t = time_step_calc(popul_num, stoch_rate, state_change_array, b, epsi)  
        if tao + delta_t > tmax:
            print("Tau leaping simulation time over")
            break
        n = np.random.normal(0, 1)  
        rxn_vector = propensity*delta_t + (propensity*delta_t)**0.5*n 
        new_popul_num = popul_num
        for j in range(len(rxn_vector)):  
            state_change_lambda = np.squeeze(np.asarray(state_change_array[j])*rxn_vector[j]) 
            new_popul_num = new_popul_num + state_change_lambda  
        if (new_popul_num < 0).any(): 
            pass 
        else: 
            popul_num = new_popul_num
            popul_num_all.append(popul_num)
            tao += delta_t 
            tao_all.append(tao)
            leap_counter += 1    
        if (popul_num < 0).any():       
            print(f"Number of molecules {popul_num} is too small for reaction to fire")
            tao_all = tao_all[0:-1]
            popul_num_all = popul_num_all[0:-1]
    print("Molecule numbers:\n", popul_num)
    print("Time of final simulation:\n", tao)
    print("leap counter:\n", leap_counter)
    print("Number of reactions:\n", len(tao_all))  
    popul_num_all = np.array(popul_num_all)
    tao_all = np.array(tao_all)
    return popul_num_all, tao_all


def parallel_func(v):
    gillespie_langevin(start_state, LHS, stoch_rate, state_change_array)

# Needs to be after the gillespie_tau_leaping function call
if __name__ == '__main__':
    start = datetime.utcnow()
    with Pool() as p:
        pool_results = p.map(parallel_func, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50])
    end = datetime.utcnow()
    sim_utc_time = end - start
    print("Simulation UTC time:\n", sim_utc_time)


def gillespie_plot(start_state, LHS, stoch_rate, state_change_array):
    """Function to plot hte results of the Gillepsie simulation"""
    popul_num_all, tao_all = gillespie_langevin(start_state, LHS, stoch_rate, state_change_array)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(tao_all, popul_num_all[:, 0], label='S', color= 'Green')
    ax1.legend()
    for i, label in enumerate(['T', 'U']):
        ax2.plot(tao_all, popul_num_all[:, i+1], label=label)
    ax2.legend()
    plt.tight_layout()
    plt.show()
    return fig


#gillespie_plot(start_state, LHS, stoch_rate, state_change_array)