import matplotlib.pyplot as plt
import numpy as np

from cmaes.cma import CMA

target = np.array([0,0])
def evaluate(x):
    n = len(x)
    if len(x) < 2:
        raise ValueError("dimension must be greater one")
    return sum([(1000 ** (i / (n - 1)) * x[i]) ** 2 for i in range(n)])
optimizer = CMA(mean=np.array([0.5, 0.5]), bounds=np.array(
    [[0, 1], [0, 1]]), sigma=0.5, n_max_resampling=1, population_size=20)
generations = 25
sqrt = int(np.sqrt(generations))
fig, axs = plt.subplots(sqrt, sqrt, num="CMA-ES", sharex=True, sharey=True)
points = np.ndarray((generations, optimizer.population_size, 2))
for g in range(generations):
    solutions = []
    for i in range(optimizer.population_size):
        point = optimizer.ask()
        points[g, i] = point
        score = evaluate(point)
        solutions.append((point, score))
    # optimizer.update_weights()
    optimizer.tell(solutions, lmm=False)

for i in range(generations):
    ax = axs[i//sqrt, i % sqrt]
    ax.scatter(*zip(*points[i]), c="b")
    ax.scatter(*target, c="r")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

plt.show()
