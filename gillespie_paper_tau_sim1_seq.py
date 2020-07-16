import numpy as np
import scipy
from scipy.special import binom
from scipy import stats
from matplotlib import pyplot as plt # allow inline plot with %matplotlib inline
import time
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Pool

# tau-leaping SSA variant of an Irriversible Isomerisam process!  
# Need to make custom class to time algorithm! 
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

        self._simulation_start_time = time.thread_time() # change clock type here
    def stop(self):
        """stop the time and report the elsaped time"""
        if self._simulation_start_time is None:
            raise TimeError(f"Timer is not running.\n Use .start() to start it.")

        self._simulation_stop_time = time.thread_time() # change clock type here
        elapsed_simulation_time = self._simulation_stop_time - self._simulation_start_time  
        self.accumulated_elapsed_time += elapsed_simulation_time

        self._simulation_start_time = None
        print(f"Elapsed time: {elapsed_simulation_time:0.10f} seconds")
    
    def get_accumulated_time(self):
        """ Return the elapsed time for later use"""
        return self.accumulated_elapsed_time

# Must make sure that all instantiated variables are not already builtin functions

# System of equations
""" S1 --> 0 == 1 --> 0
"""

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
state_change_array = np.asarray(RHS - LHS)
# state_change_array = [-1]

# Intitalise time variables
tmax = 30.0         # Maximum time
           # array to store the time of the reaction.


# function to calcualte the propensity functions for each reaction
def propensity_calc(LHS, popul_num, stoch_rate):
    """ Function to calcualte the propensity, probability of a specific reaction firing"""
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




epsi = 0.03  
def time_step_count(popul_num, stoch_rate, state_change_array, epsi): 
    """ Function to calculate the simulation time increment delta_t"""
    propensity = propensity_calc(LHS, popul_num, stoch_rate)  
    a0 = sum(propensity)
    # equation 22:   
    exptd_state_change = 0.0
    for x in range(len(propensity)):
        exptd_state_change += propensity[x]*state_change_array[x] 
    # equation 24: 
    b = 1.0 
    # equation 26: 
    numerator = epsi*a0
    denominator = abs(exptd_state_change*b)
    delta_t = numerator/denominator
    return delta_t



def gillespie_tau_leaping(initial_state, LHS, stoch_rate, state_change_array):
    """ Function to run the Gillespie stochastic simulation tau leaping variant """
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
        delta_t = time_step_count(popul_num, stoch_rate, state_change_array, epsi)
        lam = (propensity_calc(LHS, popul_num, stoch_rate)*delta_t)
        rxn_vector = np.random.poisson(lam)  
        if tao + delta_t > tmax:
            print("Tau leaping simulation time over")
            break
        if delta_t >= 2/a0:   
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
        else:
            next_t = np.random.exponential(1/a0)
            rxn_probability = propensity / a0   
            num_rxn = np.arange(rxn_probability.size)       
            if tao + next_t > tmax:      
                print("Exact SSA simulation time over")
                break
            j = stats.rv_discrete(values=(num_rxn, rxn_probability)).rvs() 
            tao = tao + next_t
            popul_num = popul_num + np.squeeze(np.asarray(state_change_array[j]))  
            popul_num_all.append(popul_num)   
            tao_all.append(tao) 
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

 

# Function that calls the Gillespie simualtion multiple times sequentially! 
def repeat_func(times, start_state, LHS, stoch_rate, state_change_array): # change function call!
    """ Function to call and run other functions multiple times """
    start = datetime.utcnow()
    for i in range(times): gillespie_tau_leaping(start_state, LHS, stoch_rate, state_change_array) 
    end = datetime.utcnow()
    print(f"Simulation UTC time:\n{end - start}")
    return i


repeat_func(10, start_state, LHS, stoch_rate, state_change_array)

print("Sequential Simualtion\nSystem 1")


def gillespie_plot(start_state, LHS, stoch_rate, state_change_array):
    """ Function to plot the results of the Gillespie simulation"""
    fig, ax = plt.subplots()
    popul_num_all, tao_all = gillespie_tau_leaping(start_state, LHS, stoch_rate, state_change_array)
    for i, label in enumerate(['S1']):
        ax.plot(tao_all, popul_num_all, label=label)
    plt.legend()
    plt.show()
    return fig


gillespie_plot(start_state, LHS, stoch_rate, state_change_array)