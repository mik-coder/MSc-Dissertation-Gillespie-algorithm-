# Implement a tau-leaping variant of the SSA   

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
tmax = 75.0        # Maximum time


# function to calcualte the propensity functions for each reaction
# Haven't checked this function so far! 
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
    # calcualte in seperate varaible then pass it into the array 
    s_derviative = stoch_rate[1]*(2*popul_num[0] -1)/2
    b = np.array([[1.0, 0.0, 0.0], [s_derviative, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.4, 0.0]])
    return b 


epsi = 0.03     # bigger epsi can get leaps to right number --> But this is the value used in the paper! 

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


# Tau-leaping while-loop method  
def gillespie_tau_leaping(initial_state, LHS, stoch_rate, state_change_array):
    """Function to implement the tau-leaping hybrid simulation
    from the gillespie paper """
    popul_num_all = [initial_state]
    popul_num = initial_state
    propensity = np.zeros(len(LHS))
    rxn_vector = np.zeros(len(LHS))
    tao = 0.0
    tao_all = [tao]
    leap_counter = 0
    while tao < tmax:
        propensity = propensity_calc(LHS, popul_num, stoch_rate)    
        a0 = (sum(propensity))  
        if a0 == 0.0:  
            print("Propensity sum is zero end execution")   
            break   
        b = update_array(popul_num, stoch_rate) 
        delta_t = time_step_calc(popul_num, stoch_rate, state_change_array, b, epsi)  
        lam = (propensity*delta_t)    
        rxn_vector = np.random.poisson(lam) 
        if tao + delta_t > tmax:
            break    
        if delta_t >= 5 / a0:   
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
        else:   # else execute the ssa because it's faster
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
def repeat_func(times, start_state, LHS, stoch_rate, state_change_array):
    """ Function to call and run other functions multiple times """
    start = datetime.utcnow()
    for i in range(times):
        popul_num_all, tao_all = gillespie_tau_leaping(start_state, LHS, stoch_rate, state_change_array) 
    end = datetime.utcnow()
    sim_time = end - start
    print("Simulation time:\n", sim_time)


repeat_func(10, start_state, LHS, stoch_rate, state_change_array)

print("Sequential\nSimulation 2")


def gillespie_plot(start_state, LHS, stoch_rate, state_change_array):
    """Function to plot hte results of the gillespie simualtion""" 
    popul_num_all, tao_all = gillespie_tau_leaping(start_state, LHS, stoch_rate, state_change_array)
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


