import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from modified_CMAES.cmaes.cma import CMA


class LMM_CMAES:
    def __init__(self, mean, sigma, population_size, n_init,seed = random.seed()):
        self._mean = mean
        self._sigma = sigma
        self._population_size = population_size
        self._n_dim = len(mean)
        self._n_b = math.floor(max(population_size / 20, 1))
        self._n_init = n_init
        self.optimizer = CMA(mean, sigma=1, n_max_resampling=1, population_size=population_size,seed=seed)
        self._lsqdim = math.floor(self._n_dim * (self._n_dim + 3) / 2 + 1)
        self._knn_target = 2 * self._lsqdim
        self._quality_threshold = (population_size ** 2) / 20
        self._mu = self._population_size // 2
        self.database = []

    def kernel(self, x):
        return (1 - x**2) ** 2

    def distance(self, x, y, M):
        return np.linalg.norm(np.matmul(M, np.subtract(x, y)))

    def is_in_database(self, point_to_search):
        for (point, score) in self.database:
            if (point_to_search == point).all():
                return True

        return False

    def update_knn(self, knn_target):
        return math.floor(min(knn_target, math.sqrt(len(self.database) * self._lsqdim)))

    def evaluate_lmm(self, point, knn):
        return self.build_metamodel(point, knn)

    def create_Z_hat(self, z):
        Z_hat = np.array([])
        for zi in z:
            Z_hat = np.append(Z_hat, zi ** 2)

        for i in range(len(z)):
            for j in range(len(z)):
                if i != j:
                    Z_hat = np.append(Z_hat, z[i] * z[j])

        for zi in z:
            Z_hat = np.append(Z_hat, zi)

        Z_hat = np.append(Z_hat, 1)

        return Z_hat

    def build_metamodel(self, q, knn):

        # Calculate matrix M
        (D, B) = np.linalg.eig(self.optimizer._C)
        D = np.diag(D)

        a = scipy.linalg.fractional_matrix_power(D, -1/2)
        M = np.matmul(a, np.transpose(B))
        M = M * (1/self._sigma)

        # sort list and take knn nearest points

        # x is under the form (point, score)
        T = sorted(self.database, key=lambda x: self.distance(np.array(x[0]), q, M))
        T = T[0:knn]

        # Calculate W (Commented out because it is not used)
        # TODO: can optimize by using the distance already calculated
        # distance_sknn = self.distance(T[knn-1][0], q, M)
        # w_aux = np.array([])
        # for (si, yi) in T:
        #     w_aux = np.append(w_aux, math.sqrt(
        #         self.kernel(self.distance(si, q, M) / distance_sknn)))

        # W = np.diag(w_aux)

        # Calculate Z, Y
        Y = np.array([])
        Z = []

        for (si, yi) in T:

            zi = np.matmul(M, np.subtract(si, q))
            Z.append(self.create_Z_hat(zi))
            Y = np.append(Y, yi)

        Z = np.array(Z)
        Y = np.array(Y)

        B_hat = np.linalg.lstsq(Z, Y.T)[0]

        # construction of A
        A = np.diag(B_hat[0:self._n_dim])
        i = 0
        j = 1
        index = self._n_dim
        while i < self._n_dim and j < self._n_dim:
            if (i != j):
                A[i][j] = B_hat[index] / 2
                index += 1
            j += 1
            if (j == self._n_dim):
                i += 1
                j = 0

        # construction of a
        a = B_hat[index: index + self._n_dim]
        a0 = B_hat[-1]

        return a0

    def build_database(self, evaluate):
        for _ in range(self._lsqdim + 1):
            point = self.optimizer.ask()
            score = evaluate(point)
            self.database.append((point, score))

    def evalutae_lmm(self, point, knn):
        return self.build_metamodel(point, knn)


    # Return number of calls to evaluate or -1 if maximum reached
    def run(self, evaluate, maximum_evaluations=100):

        # sqrt = int(np.sqrt(maximum_evaluations))
        # fig, axs = plt.subplots(sqrt, sqrt, num="LMM CMA-ES", sharex=True, sharey=True)
        # plot_points = np.ndarray((maximum_evaluations, self._population_size, 2))

        # build database
        self.build_database(evaluate)

        counter_evaluations = 0
        counter_generations = 0

        # Start the optimization
        while counter_evaluations < maximum_evaluations:
            # first iteration
            knn = self.update_knn(self._knn_target)
            solutions = []
            xks = []
            # for every xk
            check_all_point = True
            for k in range(self._population_size):
                point = self.optimizer.ask()
                # plot
                # plot_points[counter_generations, k] = point
                if abs(evaluate(point)) > 10**-7:
                    check_all_point = False

                score_lmm = self.evaluate_lmm(point, knn)
                solutions.append((point, score_lmm))
                xks.append(point)

            if check_all_point == True:
                print("LMM_CMAES: Error reached")
                print("evaluations: ", counter_evaluations)
                print("Generations: ", counter_generations)

                break

            # Rank the solutions
            solutions.sort(key=lambda tup: tup[1])
            rank = solutions[:self._mu]

            # Add n_init points and their corresponding f(point) to the database
            for (point, _) in rank[:self._n_init]:
                score = evaluate(point)

                counter_evaluations += 1
                self.database.append((point, score))

            # Second iteration
            knn = self.update_knn(self._knn_target)
            solutions = []
            for point in xks:
                score_lmm = self.evalutae_lmm(point, knn)
                solutions.append((point, score_lmm))
            print(solutions)

            # Rank the solutions
            solutions.sort(key=lambda tup: tup[1])
            rank = solutions[:self._mu]

            model_error = math.inf
            counter = 0

            while counter < self._population_size and model_error > self._quality_threshold:
                counter = counter + 1
                j = 0
                old_rank = copy.deepcopy(rank)
                while j < self._mu:
                    point_to_search = rank[j][0]
                    if not self.is_in_database(point_to_search):
                        # Evaluate the next n_b best unevaluated individuals of the previous rank
                        score = evaluate(point_to_search)
                        counter_evaluations += 1

                        self.database.append((point_to_search, score))
                    j += 1
                    if j == self._n_b:
                        break

                knn = self.update_knn(self._knn_target)
                solutions = []
                for point in xks:
                    score_lmm = self.evalutae_lmm(point, knn)
                    solutions.append((point, score_lmm))
                # Rank
                solutions.sort(key=lambda tup: tup[1])
                rank = solutions[:self._mu]

                model_error = 0
                for g in range(self._mu):
                    model_error += abs(old_rank[g][1] - rank[g][1])

                rank = solutions

            if counter > 2:
                self._n_init = min(self._population_size, self._n_init + self._n_b)
            if counter < 2:
                self._n_init = max(0, self._n_init - self._n_b)
            
            self.optimizer.update_weights()
            self.optimizer.tell(rank, lmm=True)

            counter_generations += 1
            

        if (counter_evaluations == maximum_evaluations):
            print("LMM_CMAES: Maximum evaluations reached")
            return -1
            
        # for i in range(counter_evaluations):
        #     ax = axs[i//sqrt, i % sqrt]
        #     ax.scatter(*zip(*plot_points[i]), c="b")
        #     # ax.scatter(*np.array([0, 0]), c="r")
        #     ax.set_xlim([-5, 5])
        #     ax.set_ylim([-5, 5])

        # plt.show()
        return counter_evaluations
