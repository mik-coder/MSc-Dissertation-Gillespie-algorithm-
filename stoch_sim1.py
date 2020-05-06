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
popul_num = np.array([200, 100, 0, 0])             

# The SSA implementation
# STEP 1
# Intialise all the parameters

# Define the stochiometry of the system as a numpy array
# matrix of the reactants stochiometry (Left hand side) of 
# each equation in the system
# need to add padding to array!
LHS = np.array([[1,1,0,0], [0,0,1,0], [0,0,1,0]])
print("Reactant stochiometry matrix:\n ", LHS)

# matrix of the product stochiometry (Right hand side) of 
# each equation in the system
RHS = np.matrix([[0,0,1,0], [1,1,0,0], [1,0,0,1]])
print("Product stochiometry matrix:\n ", RHS)

# number of specieis in model
""" E, S, ES, P """
num_spec = 4

# Define stochastic rate parameters
stoch_rate = np.array([0.0016, 0.0001, 0.1000])      
print(type(stoch_rate)) 
print("The stochastic rate constants, respectivly:\n ", stoch_rate)

# Define the state change vector
state_change_matrix = RHS - LHS
print("The state change matrix is: \n", state_change_matrix)

# Intitalise time variables 
tmax = 100.0         # Maximum time 
tao = 0.0           # array to store the time of the reaction.   
# missing required argument shape

# STEP 3
# Define a function to calculate the propensity functions
propensity = np.zeros(len(LHS))

def propensity_calc(LHS, popul_num, stoch_rate):
    for row in range(len(LHS)):     
            a = stoch_rate[row]        
            for i in range(len(popul_num)):   
                if (popul_num[i] >= LHS[row, i]).all():       # seems to work --> iterates through multiple elementa      
                    # will return a new array/ value with just true or false --> How to use this further
                    binom_rxn = (binom(popul_num[i], LHS[row, i]))    
                    a = a*binom_rxn            
                else: 
                    a = 0   
                    break 
            propensity[row] = a 
    return propensity.astype(float)


print(propensity_calc(LHS, popul_num, stoch_rate))

# using .any() returns type and value error
# using .all runs without error --> But doesnt work quite right! 



# Calculate sum of propensity fuctions 
a0 = sum(propensity)
print(a0)


# Calculate the probability of each reaction firing as a fraction of propensity/a0
def prob_rxn_fires(propensity, a0):
    prob = propensity/a0   
    return prob 


rxn_probability = (prob_rxn_fires(propensity, a0))
print("probability of reaction:\n", rxn_probability)

# create new array to hold the number of reactions in the system
# must be the same shape and size of rxn_probabiltiy for rv_discrete to work
num_rxn = np.arange(1, rxn_probability.size + 1).reshape(rxn_probability.shape)
print(num_rxn)      

# SSA while loop
while tao < tmax: 
    propensity_calc(LHS, popul_num, stoch_rate)
    a0 = sum(propensity)
    rxn_probability = propensity / a0
    print(rxn_probability)
    num_rxn = np.arange(1, rxn_probability.size + 1).reshape(rxn_probability.shape)
    if a0 < 0:     # breaks prematurely here with <= but works fine with < ????
        break
    else:
        t = np.random.exponential(a0)   
        print("time of next reaction:\n", t)   
        # sample time system stays in given state from exponential distribution of the propensity sum. 
        if tao + t > tmax: 
            print("total time of simulation:\n", tao)
            tao = tmax
            break
    j = stats.rv_discrete(name="Reaction index", values=(num_rxn, rxn_probability).rvs()
    new_tao = tao + t   # two different types of tao? One is int the other
    print("tao:\n", tao)     
    popul_num = popul_num + state_change_matrix[j]    # update state of system/ popul_num
    # j = 1! When using it to index row of state_change_matrix --> always index the SECOND row! 
    # is add SECOND row of state change matrix [1, 1, -1, 0]
    # and not the FIRST row of state change matrix [-1, -1, 1, 0]
    # need to iterate over whole of state change matrix ???  

    # not sampling the next reaction (j = rv_discrete...) properly
    # therefore not updating the popul_num properly when using state_change_matrix[j]
    print("New popul_numbers:\n", popul_num)
 



 
    # Then error messages: ValueError: The sum of provided pk is not 1.
    # pk = propensity/a0 --> list of probabilities that must add up to one! 
    # Try evaluate propensity/a0 INSIDE the while loop



# Need to calculate the propensities in the loop so they can be re-evaluated after each iteration! 

# tao is initialised outside of the while loop to zero 
# first itration runs
# time of first reaction is sampled from exponential and assigned to variable t
# tao = tao + t --> tao = 0 + t
# second iteration runs
# time of second reaction is sampled from exponential
# result == 0 ????
# why is time of next reaction 0 after first iteration ? 


# state change matrix is correct!
# but is adding not subtracting values onto popul_num

# subtracting state matrix makes it through first iteration
# and then causes the following errors: 
# TypeError: only size-1 arrays can be converted to Python scalars
# The above exception was the direct cause of the following exception:
# ValueError: setting an array element with a sequence.
# ????

# Plotting the output!  
plt.plot(popul_num)      
plt.show()





# chat link https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.exponential.html


# other links 
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html?highlight=discrete%20random

#https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html?highlight=exponential#numpy.random.exponential

#https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.exponential.html
   
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html
