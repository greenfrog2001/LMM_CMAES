import random
import cmaes
import numpy as np

# Return number of calls to evaluate or -1 if maximum reached
def run(evaluate, mean, sigma, population_size,maximum_evaluation, seed):
    counter_evaluation = 0
    optimizer = cmaes.CMA(mean=mean, sigma=sigma,
                          population_size=population_size, seed=seed)
    counter_generations = 0
    while counter_evaluation < maximum_evaluation:
        solutions = []
        check_all_point = True

        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = evaluate(x)
            counter_evaluation += 1
            if (counter_evaluation > maximum_evaluation):
                return None
            if abs(value - 0) > 1.e-7:
                check_all_point = False

            solutions.append((x,value))

        if check_all_point == True:
            print("CMAES: Error reached")
            print("evaluations: ", counter_evaluation)
            print("generations: ", counter_generations)
            return counter_evaluation
        optimizer.tell(solutions)
        counter_generations += 1
        
    if (counter_evaluation == maximum_evaluation):
        print("LMM_CMAES: Maximum evaluations reached")  
    return -1
        


