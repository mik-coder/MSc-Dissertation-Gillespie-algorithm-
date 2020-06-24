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
import multiprocessing as mp
from multiprocessing import Pool
import os
#import cProfile
#import pstats
#import io

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
popul_num = np.array([2.0E5, 1.0E5, 0, 0])

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
                    binom_rxn = binom(popul_num[i], LHS[row, i])    # can return v large numbers
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

epsi = 0.03

# returns the value to increment time by --> delta_t
def time_step_calc(propensity_calc, state_change_array, b, epsi):
    """ Function to calculate the simulation 
    time increment delta_t"""
    propensity = propensity_calc(LHS, popul_num, stoch_rate) 
    denominator = np.zeros(len(propensity))
    a0 = sum(propensity)
    # equation 22:
    exptd_state_array = 0.0
    for x in range(len(propensity)):
        exptd_state_array +=  propensity[x]*state_change_array[x]
        # expected states for first two reactions are very large
    # equation 24: Calculated ad-hoc results in bji matrix
    numerator = epsi*a0 
    for j in range(len(propensity)):
        for i in range(len(popul_num)):
            denominator[j] += (exptd_state_array[i]*b[j, i])
    # equation 26
    delta_t_array = numerator/abs(denominator)
    delta_t = min(delta_t_array)
    return delta_t


# Tau-leaping  method
popul_num_all = [popul_num]
tao_all = [tao]
propensity = np.zeros(len(LHS))
rxn_vector = np.zeros(len(LHS))  


def gillespie_tau_leaping(propensity_calc, popul_num, LHS, stoch_rate, popul_num_all, tao_all, rxn_vector, tao, epsi): 
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
        b = update_array(popul_num, stoch_rate)
        delta_t = time_step_calc(propensity_calc, state_change_array, b, epsi)
        lam = (propensity_calc(LHS, popul_num, stoch_rate)*delta_t)
        rxn_vector = np.random.poisson(lam) 
        if tao + delta_t > tmax:
            break
        if delta_t >= 1/a0:
            tao += delta_t 
            #print("tao:\n", tao)
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
    print("Number of reactions:\n", len(tao_all))
    print("leap counter:\n", leap_counter)
    t.stop()
    return popul_num_all.append(popul_num), tao_all.append(tao)


gillespie_tau_leaping(propensity_calc, popul_num, LHS, stoch_rate, popul_num_all, tao_all, rxn_vector, tao, epsi)


if __name__ == '__main__':
    t = simulation_timer()
    t.start()
    with Pool() as pool:
        result = pool.map_async(gillespie_tau_leaping, (propensity_calc, popul_num, LHS, stoch_rate, popul_num_all, tao_all, rxn_vector, tao, epsi))
        print(result.get())
        print(pool.map(gillespie_tau_leaping, (propensity_calc, popul_num, LHS, stoch_rate, popul_num_all, tao_all, rxn_vector, tao, epsi)))
        t.stop()


# Doing parallel processing times out my funcions much quicker why?

# calculates the time elsaped for each indivdual process instead of all 4 --> which is longer than just running one!
# ^^^Does work! Runs 4 simulations but then hits an error
# some sort of error in this line try _async equivalents for pool.map and pool.apply


# Plotting the graph 
popul_num_all = np.array(popul_num_all)
tao_all = np.array(tao_all)

for i, label in enumerate(['Enzyme', 'Substrate', 'Enzyme-Substrate complex', 'Product']):
    plt.plot(tao_all, popul_num_all[:, i], label=label)
plt.legend()
plt.tight_layout()
plt.show()



# parallelisation code: 
# Run a specified number of simulations in parallel
# How many processes can I run? 
#print("Number of processors: ", mp.cpu_count())





#if __name__ == '__main__':
#    jobs = []
#    for i in range(4):
#        p = mp.Process(target=gillespie_tau_leaping, args=(propensity_calc, popul_num, LHS, stoch_rate, popul_num_all, tao_all, rxn_vector, tao, epsi))
#        jobs.append(p)
#        p.start()
# seems to run processes one after the other instead of in parallel
# also throws attribute errors after the first run why?!?! 
# some of the graphs are wrong some aren't??? 



#if __name__ == '__main__':
#    with Pool() as Pool:   
#        result = Pool.apply_async(gillespie_tau_leaping, args=(propensity_calc, popul_num, LHS, stoch_rate, popul_num_all, tao_all, rxn_vector, tao, epsi))  
#        print(result.get())     # leaving .get() allows it to take as long as it needs and prevents TimeoutError.

# Really tempramental!
# Sometimes it seems to work and somethimes it doesnt! 



# ^^^What does this mean?^^^
#multiple_results = [Pool.apply_async(os.getpid, ()) for i in range(3)]
#print([res.get() for res in multiple_results])

# Runs --> produces one correct graph
# Then enters parallel part
#   prints one incorrect graph 
#       Doesn't time out or throw error 
#           Doesn't print the right number of processes? --> Not sure what it returns? 



#if __name__ == '__main__':
#    p = mp.Process(target=gillespie_tau_leaping, args=(propensity_calc, popul_num, LHS, stoch_rate, popul_num_all, tao_all, rxn_vector, tao, epsi))
#    p.start()


# runs simualtion with the wrong graph
# then it runs and returns a correct graph 
# then it crashes with AttributeError: 'numpy.ndarray' object has no attribute 'append'


 
# code for profiling 
#profile = cProfile.Profile()
#profile.enable()
"""function here"""
#profile.disable()
#s = io.StringIO()
#ps = pstats.Stats(profile, stream=s).sort_stats('tottime')
#ps.print_stats()
#with open('tau_ssa_MM_performance.txt', 'w+') as f:
#    f.write(s.getvalue())