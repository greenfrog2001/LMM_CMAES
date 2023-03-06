import csv
import random
from random import seed
import matplotlib.pyplot as plt

import numpy as np

import Cmaes_run
import LMM_CMAES
import functions


# open the file in the write mode
f = open('result.csv', 'w')

# create the csv writer
writer = csv.writer(f)
writer.writerow(["dimension","LMM_CMAES","CMAES"] )

# Maximum number of iterations
max_iterations = 1

# Maximum number of evaluations
maximum_evaluation = 1000000

# Maximum of dimension
maximum_dimension = 3

# Population size
population_size = 20

# Sigma
sigma = 1

range_dimension  = range(2, maximum_dimension + 1)

mean_lmm = []
mean_cmaes = []

for dim in range_dimension:
    result_lmm = []
    result_cmaes = []
    temp1 = 0
    temp2 = 0
    for i in range(max_iterations):
        print("\nIteration: ", i, "\n")
        seed = random.seed()
        LMM_CMAES_RUN = LMM_CMAES.LMM_CMAES(mean=np.array(
            [2.0] * dim),  sigma=sigma, population_size=population_size,  n_init=1, seed=seed)
        a = LMM_CMAES_RUN.run(functions.Cigar, maximum_evaluations=maximum_evaluation)
        if (a != -1):
            temp1 += a

        a = temp2 + Cmaes_run.run(functions.Cigar, mean=np.array(
            [2.0]*dim), sigma=sigma, population_size=population_size, maximum_evaluation=maximum_evaluation, seed=seed)
        if (a != -1):
            temp2 +=a
        

        result_lmm.append(temp1)
        result_cmaes.append(temp2)

    mean_lmm.append(temp1/len(result_lmm))
    mean_cmaes.append(temp2/len(result_cmaes))
    
    writer.writerow([dim,temp1/len(result_lmm), temp2/len(result_cmaes)])


plt.plot(range_dimension, mean_lmm, label = "LMM_CMAES")
plt.plot(range_dimension, mean_cmaes, label = "CMAES")
plt.xlabel("Dimension")
plt.ylabel("Average number of evaluations")
plt.legend()
plt.savefig('result.png')
plt.show()
    

f.close()