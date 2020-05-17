# https://github.com/karinsasaki/gillespie-algorithm-python/blob/master/build_your_own_gillespie_exercises.ipynb
# ^^^ Github example of stochastic implementation ^^^
# http://be150.caltech.edu/2016/handouts/gillespie_simulation.html
# ^^^ Example algorithm ^^^
import numpy as np
import scipy
from scipy.special import binom
from scipy import stats
from matplotlib import pyplot as plt # allow inline plot with %matplotlib inline

# System of equations
"""
E + S --> ES == 1E + 1S + 0ES + 0P --> 0E + 0S + 1ES + 0P
ES --> E + S == 0E + 0S + 1ES + 0P --> 1E + 1S + 0ES + 0P
ES --> E + P == 0E + 0S + 1ES + 0P --> 1E + 0S + 0ES + 1P
"""
# Need to initialise the discrete numbers of molecules???
popul_num = np.array([700, 350, 0, 0])

# ratios of starting materials for each reaction 
LHS = np.array([[1,1,0,0], [0,0,1,0], [0,0,1,0]])

# ratios of products for each reaction
RHS = np.matrix([[0,0,1,0], [1,1,0,0], [1,0,0,1]])

# stochastic rates of reaction
stoch_rate = np.array([0.0016, 0.0001, 0.1])

# Define the state change vector
state_change_array = RHS - LHS

# Intitalise time variables
tmax = 100         # Maximum time
tao = 0.0           # array to store the time of the reaction.

def propensity_calc(LHS, popul_num, stoch_rate):
    propensity = np.zeros(len(LHS))
    for row in range(len(LHS)):
            a = stoch_rate[row]     # type = numpy.float64
            for i in range(len(popul_num)):
                if (popul_num[i] >= LHS[row, i]):       # seems to work --> iterates through multiple elementa
                    # will return a new array/ value with just true or false --> How to use this further
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
    tao = tao + t
    popul_num = popul_num + np.squeeze(np.asarray(state_change_array[j]))   # add state change array for that reaction!
    print("state change array:\n", state_change_array[j])
    popul_num_all.append(popul_num) 


popul_num_all = np.array(popul_num_all)
for i, label in enumerate(['Enzyme', 'Substrate', 'Enzyme-Substrate complex', 'Product']):
    plt.plot(popul_num_all[:, i], label=label)
plt.legend()
plt.tight_layout()
plt.show()