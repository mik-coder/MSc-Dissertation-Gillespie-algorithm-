# https://github.com/karinsasaki/gillespie-algorithm-python/blob/master/build_your_own_gillespie_exercises.ipynb
# ^^^ Github example of stochastic implementation ^^^
# http://be150.caltech.edu/2016/handouts/gillespie_simulation.html
# ^^^ Example algorithm ^^^
import numpy as np
import scipy
from scipy.special import binom
from scipy import stats
from matplotlib import pyplot as plt # allow inline plot with %matplotlib inline
import time

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
        # should stop the timer!
        elasped_simulation_time = self._simulation_stop_time - self._simulation_start_time  
        self._simulation_start_time = None
        print(f"Elasped time: {elasped_simulation_time:0.4f} seconds")


# Need to initialise the discrete numbers of molecules???
popul_num = np.array([1.0E5, 0, 0])

# ratios of starting materials for each reaction 
LHS = np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 1, 0]])

# ratios of products for each reaction
RHS = np.array([[0, 0, 0], [0, 1, 0], [2, 0, 0], [0, 0, 1]])

# stochastic rates of reaction
stoch_rate = np.array([1.0, 0.002, 0.5, 0.04])

# Define the state change vector
state_change_array = np.asarray(RHS - LHS)  
print("State change array", state_change_array)   

# Intitalise time variables
tmax = 30.0         # Maximum time
tao = 0.0           # array to store the time of the reaction.

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
            propensity[row] = a     # type = numpy.ndarray
    return propensity

popul_num_all = [popul_num]
tao_all = [tao]

propensity = np.zeros(len(LHS))
def gillespie_exact_ssa(propensity_calc, popul_num, popul_num_all, tao_all, tmax, tao):
    t = simulation_timer()
    t.start()
    while tao < tmax:
        propensity = propensity_calc(LHS, popul_num, stoch_rate)
        a0 = (sum(propensity))
        if a0 == 0.0:
            break
        next_t = np.random.exponential(1/a0)     # sampling time of next reaction here --> time smaller ever time when rate is bigger! 
        rxn_probability = propensity / a0   
        num_rxn = np.arange(rxn_probability.size)       
        if tao + next_t > tmax:
            tao = tmax
            break
        j = stats.rv_discrete(values=(num_rxn, rxn_probability)).rvs()      # sampling next reaction to occur here
        tao = tao + next_t
        print("tao:\n", tao)
        popul_num = popul_num + np.squeeze(np.asarray(state_change_array[j]))   # add state change array for that reaction!
        print("Molecule numbers:\n", popul_num)
        popul_num_all.append(popul_num) 
        tao_all.append(tao)
    t.stop()
    return popul_num_all.append(popul_num), tao_all.append(tao)


print(gillespie_exact_ssa(propensity_calc, popul_num, popul_num_all, tao_all, tmax, tao))


# plotting
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



# with popul_num[i] shows all three species being used at the same rate??? 
# without ledgend shows three lines for each species but only the values of U are plotted? 

# three lines for each speicies but only plots the ones assigned U 
# Plotted lines for U aren't right --> show it being used up or not produced!  
# 