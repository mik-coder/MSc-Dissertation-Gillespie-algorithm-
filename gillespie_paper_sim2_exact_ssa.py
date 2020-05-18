# https://github.com/karinsasaki/gillespie-algorithm-python/blob/master/build_your_own_gillespie_exercises.ipynb
# ^^^ Github example of stochastic implementation ^^^
# http://be150.caltech.edu/2016/handouts/gillespie_simulation.html
# ^^^ Example algorithm ^^^
import numpy as np
import scipy
from scipy.special import binom
from scipy import stats
from matplotlib import pyplot as plt # allow inline plot with %matplotlib inline

# Exact SSA of an Irriversible Isomerisam process!  

# System of equations
# stochiometric equation needs to have equal length! 
""" 1S + 0T + 0U --> 0S + 0T + 0U
    2S + 0T + 0U --> 0S + 1T + 0U 
    0S + 1T + 0U --> 2S + 0T + 0U
    0S + 1T + 0U --> 0S + 0T + 1U   
"""

# Need to initialise the discrete numbers of molecules???
popul_num = np.array([1.0E5, 0, 0])

# ratios of starting materials for each reaction 
LHS = np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 1, 0]])

# ratios of products for each reaction
RHS = np.matrix([[0, 0, 0], [0, 1, 0], [2, 0, 0], [0, 0, 1]])

# stochastic rates of reaction
stoch_rate = np.array([1.0, 0.002, 0.5, 0.04])

# Define the state change vector
state_change_array = RHS - LHS     

# Intitalise time variables
tmax = 20.0         # Maximum time
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

propensity = np.zeros(len(LHS))
while tao < tmax:
    propensity = propensity_calc(LHS, popul_num, stoch_rate)
    a0 = (sum(propensity))
    if a0 == 0.0:
        break
    t = np.random.exponential(1/a0)
    rxn_probability = propensity / a0   
    num_rxn = np.arange(rxn_probability.size)       
    if tao + t > tmax:
        tao = tmax
        break
    j = stats.rv_discrete(values=(num_rxn, rxn_probability)).rvs()
    # sampling next reaction j --> something here to ensure 100,000 reactions are simulated/sampled? 
    tao = tao + t
    popul_num = popul_num + np.squeeze(np.asarray(state_change_array[j]))   # add state change array for that reaction!
    print("Population numbers:\n", popul_num)
    print("Simulation time:\n", t, tao)
    popul_num_all.append(popul_num) 

# How to ensure that 100,000 reactions are simulated
# That a state is plotted only after every 800 reactions? 

popul_num_all = np.array(popul_num_all)


for i, label in enumerate(['S', 'T', 'U']):
    plt.plot(popul_num_all[i], label=label)   # removing the [i] index plots a LINEAR decay --> Gillespie plot is a CURVED decay
plt.legend()
plt.tight_layout()
plt.show()