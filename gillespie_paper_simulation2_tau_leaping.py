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
        print(f"Elasped time: {elasped_simulation_time:0.4f} seconds")

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
                    aj = a*binom_rxn
                else:
                    aj = 0
                    break
            propensity[row] = aj    
    return propensity


epsi = 0.03
popul_num_all = [popul_num]
tao_all = [tao]
propensity = np.zeros(len(LHS))
rxn_vector = np.zeros(len(LHS))  
a0 = sum(propensity)

 
def update_matrix(popul_num, stoch_rate): 
    """Specific to this model 
    will need to change if different model 
    implements equaiton 24 of the Gillespie paper"""
    b = np.array([[1.0, 0.0, 0.0], [stoch_rate[1]*(2*(popul_num[0]) - 1)/2, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.4, 0.0]])
    return b
b = update_matrix(popul_num, stoch_rate)


def time_step_calc(propensity_calc, state_change_array, b, epsi):
    """ Function to calculate the simulation 
    time increment delta_t"""
    evaluate_propensity = propensity_calc(LHS, popul_num, stoch_rate) 
    denominator = np.zeros(len(evaluate_propensity))
    a0 = sum(evaluate_propensity)
    # equation 22:
    exptd_state_array = 0.0
    for x in range(len(evaluate_propensity)):
        exptd_state_array +=  evaluate_propensity[x]*state_change_array[x]
    print("expectd state array:\n", exptd_state_array) 
    # equation 24: Calculated ad-hoc results in bji matrix
    numerator = epsi*a0 
    for j in range(len(evaluate_propensity)):
        for i in range(len(popul_num)):
            denominator[j] += (exptd_state_array[i]*b[j, i])
    # equation 26
    delta_t_array = numerator/abs(denominator)
    print("delta_t_array:\n", delta_t_array)
    delta_t = min(delta_t_array)
    print("The calculated value of delta_t:\n", delta_t)
    return delta_t
delta_t = time_step_calc(propensity_calc, state_change_array, b, epsi)



# Tau-leaping while-loop method


def gillespie_tau_leaping(propensity_calc, popul_num, popul_num_all, tao_all, rxn_vector, delta_t, tao, epsi):
    t = simulation_timer()
    t.start()
    while tao < tmax:
        evaluate_propensity = propensity_calc(LHS, popul_num, stoch_rate)     
        a0 = (sum(evaluate_propensity))  # a0 is a numpy.float64 
        if a0 == 0.0:      
            break   
        if popul_num.any() < 0:      # Not working!
            break    
        lam = (evaluate_propensity*delta_t)     # still uses evaluate propensity? 
        rxn_vector = np.random.poisson(lam) # probability of a reaction firing in the given time period
        #print("reaction vector:\n", rxn_vector)
        if tao + delta_t > tmax:
            break    
        tao += delta_t 
        leap_count = tao / delta_t
        # Calculate the number of leaps      
        # Not even doing the tao-leap variant! Goes straight to the exact ssa
        if delta_t >= 1/ a0:
            for j in range(len(rxn_vector)):  
                state_change_lambda = np.squeeze(np.asarray(state_change_array[j])*rxn_vector[j]) 
                #print("state_change_lambda:\n", state_change_lambda)
                # only doing the first reaction ? Is it updating the system?  
            popul_num = popul_num + state_change_lambda                
            popul_num_all.append(popul_num)
            tao_all.append(tao) 
            print("Leap count:\n", leap_count)
        else:
            next_t = np.random.exponential(1/a0)
            rxn_probability = evaluate_propensity / a0   
            num_rxn = np.arange(rxn_probability.size)       
            if tao + next_t > tmax:      
                tao = tmax
                break
            j = stats.rv_discrete(values=(num_rxn, rxn_probability)).rvs() # sum of pk provided is not 1?!
            tao = tao + next_t
            popul_num = popul_num + np.squeeze(np.asarray(state_change_array[j]))  
            popul_num_all.append(popul_num)   
            tao_all.append(tao)    
    print("Molecule numbers:\n", popul_num)
    print("Time of final simulation:\n", tao)
    t.stop()
    return popul_num_all.append(popul_num), tao_all.append(tao)


print(gillespie_tau_leaping(propensity_calc, popul_num, popul_num_all, tao_all, rxn_vector, delta_t, tao, epsi))

# returns blank plots!! 


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


