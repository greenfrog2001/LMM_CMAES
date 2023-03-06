import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np

import meta_model
from cmaes.cma import CMA

target = np.array([0.75, 0.75])
def update_knn(knn_target, database, lsqdim):
    return math.floor(min(knn_target, math.sqrt(len(database) * lsqdim)))



# Initialization
D = 2  # the dimension of each point x_k
lambda_ = 20  # not mentioned in the paper???
lsqdim = math.floor(D * (D + 3) / 2 + 1)
quality_threshold = lambda_ * lambda_ / 20
n_init = 1
n_b = math.floor(max(lambda_/20, 1))
knn_target = 2 * lsqdim
mu = math.floor(lambda_ / 2)  # not mentioned in the paper?

optimizer = CMA(mean=np.array([2.0,2.0]) , sigma=1, n_max_resampling=1, population_size=lambda_)
generations = 25

## plot
sqrt = int(np.sqrt(generations))
fig, axs = plt.subplots(sqrt, sqrt, num="LMM CMA-ES", sharex=True, sharey=True)
points = np.ndarray((generations, optimizer.population_size, 2))

def evaluate(x):
    n = len(x)
    if len(x) < 2:
        raise ValueError("dimension must be greater one")
    return sum([(1000 ** (i / (n - 1)) * x[i]) ** 2 for i in range(n)])


def is_in_database(point_to_search, database):
    for (point, score) in database:
        if (point_to_search == point).all():
            return True

    return False


def evalutae_lmm(optimizer, point, database, knn):
    return meta_model.build_metamodel(database, point, optimizer._C, optimizer._sigma, knn, optimizer._n_dim)


# Building database S
database = []
#evaluated_points = []
for i in range(lsqdim + 1):
    point = optimizer.ask()
    score = evaluate(point)
    database.append((point, score))

    # evaluated_points.append(list(point))


#plot
points = np.ndarray((generations, optimizer.population_size, 2))

for i in range(generations):
    # First iter
    knn = update_knn(knn_target, database, lsqdim)
    solutions = []
    xks = []
    # for every xk
    for k in range(lambda_):
        point = optimizer.ask()
        points[i, k] = point
        score_lmm = evalutae_lmm(optimizer, point, database, knn)
        solutions.append((point, score_lmm))
        xks.append(point)

    # Rank the solutions
    solutions.sort(key=lambda tup: tup[1])

    rank = solutions[:mu]
    # Add n_init points and their corresponding f(point) to the database
    for (point, _) in rank[:n_init]:
        score = evaluate(point)
        database.append((point, score))
        # evaluated_points.append(list(point))

    # Second iter
    knn = update_knn(knn_target, database, lsqdim)
    solutions = []
    for point in xks:
        score_lmm = evalutae_lmm(optimizer, point, database, knn)
        solutions.append((point, score_lmm))

    # Rank the solutions
    solutions.sort(key=lambda tup: tup[1])
    rank = solutions[:mu]

    model_error = math.inf
    counter = 0

    while counter < lambda_ and model_error > quality_threshold:  # Lack one condition!!!
        counter = counter + 1
        j = 0
        old_rank = copy.deepcopy(rank)
        while j < mu:
            point_to_search = rank[j][0]
            if not is_in_database(point_to_search, database):
                # Evaluate the next n_b best unevaluated individuals of the previous rank
                score = evaluate(point_to_search)
                database.append((point_to_search, score))
            j += 1
            if j == n_b:
                break

        knn = update_knn(knn_target, database, lsqdim)
        solutions = []
        for point in xks:
            score_lmm = evalutae_lmm(optimizer, point, database, knn)
            solutions.append((point, score_lmm))
        # Rank
        solutions.sort(key=lambda tup: tup[1])
        rank = solutions[:mu]

        model_error = 0
        for g in range(mu):
            model_error += abs(old_rank[g][1] - rank[g][1])

        rank = solutions

    if counter > 2:
        n_init = min(lambda_, n_init + n_b)
    if counter < 2:
        n_init = max(0, n_init - n_b)

    optimizer.update_weights()
    optimizer.tell(rank, lmm=True)
    print("Generation: ", i, "database ", database, "\n")

for i in range(generations):
    ax = axs[i//sqrt, i % sqrt]
    ax.scatter(*zip(*points[i]), c="b")
    ax.scatter(*target, c="r")
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

plt.show()
