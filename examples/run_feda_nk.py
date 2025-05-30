# feda_project/examples/run_feda_nk.py
import sys
import os
# Ensure the project root is in the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) 
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


import numpy as np
import matplotlib.pyplot as plt 
# Updated imports for separate files
from feda_algorithm.rf_mimic import RF_MIMIC
from feda_algorithm.mimic_o2 import MIMIC_O2
from feda_algorithm.mimic_my import MIMIC_MY # Added MIMIC_MY import
from problem_definitions.nk_landscape import NKLandscapeProblem
from utils.debugging import print_debug, DEBUG_MODE

# To see debug messages from the algorithm, set DEBUG_MODE to True in utils/debugging.py
# import utils.debugging
# utils.debugging.DEBUG_MODE = True # Uncomment to enable debug prints

if __name__ == "__main__":
    print("Comparing RF_MIMIC, MIMIC_O2, and MIMIC_MY on NK-Landscape Problem...")

    # 1. Define the problem (once for all algorithms)
    N_genes = 50
    K_interactions = 20 # User updated K
    landscape_seed = 40   # User updated seed
    problem = NKLandscapeProblem(N=N_genes, K=K_interactions, landscape_seed=landscape_seed)
    print(f"Problem: {problem} with landscape_seed={landscape_seed}")

    algorithms_to_run = [
        {
            "name": "RF_MIMIC",
            "class": RF_MIMIC,
            "config": {
                "problem": problem,
                "population_size": 3000,
                "max_iterations": 50,
                "elite_ratio": 0.25,
                "rf_params": {'n_estimators': 200, 'min_samples_leaf': 2, 'max_depth': 25},
                "branch_alpha": 0.05
            }
        },
        {
            "name": "MIMIC_O2",
            "class": MIMIC_O2,
            "config": {
                "problem": problem,
                "population_size": 3000,
                "max_iterations": 50,
                "elite_ratio": 0.2 
            }
        },
        { # ADDED: Configuration for MIMIC_MY
            "name": "MIMIC_MY",
            "class": MIMIC_MY,
            "config": {
                "problem": problem,
                "population_size": 3000, # Example population size, same as MIMIC_O2 for this run
                "max_iterations": 50,
                "elite_ratio": 0.2  # Using standardized name
            }
        }
    ]

    results_summary = []
    run_random_seed = 123 

    plt.figure(figsize=(9, 6)) 

    # Loop through and run each algorithm
    for algo_info in algorithms_to_run:
        algo_name = algo_info["name"]
        AlgoClass = algo_info["class"]
        config = algo_info["config"]

        print(f"\n--- Running Algorithm: {algo_name} ---")
        print(f"Configuring {algo_name} optimizer...")
        optimizer = AlgoClass(**config)
        
        print(f"Optimizer configured with pop_size={optimizer.population_size}, max_iter={optimizer.max_iterations}, elite_ratio={optimizer.elite_ratio}")

        print(f"\nStarting optimization for {algo_name}...")
        best_solution, best_fitness, fitness_history, avg_fitness_history, timing_info = optimizer.run(random_seed=run_random_seed)

        print(f"\n--- {algo_name} Optimization Complete ---")
        total_time = timing_info.get('total_time', -1.0)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Best fitness achieved: {best_fitness:.4f}")
        if best_solution is not None:
            print(f"Best solution found (first 10 genes): {best_solution[:10]}...")
        else:
            print("No solution found.")
        
        results_summary.append({
            "name": algo_name,
            "best_fitness": best_fitness,
            "total_time": total_time,
            "fitness_history": fitness_history, 
            "avg_fitness_history": avg_fitness_history 
        })

        if avg_fitness_history:
            # Ensure iterations match length of avg_fitness_history which includes initial state
            iterations = range(len(avg_fitness_history)) 
            plt.plot(iterations, avg_fitness_history, label=f'{algo_name} Avg Fitness')

    plt.title(f'Average Population Fitness vs. Iteration (N={N_genes}, K={K_interactions})')
    plt.xlabel('Iteration (0 = Initial Population)')
    plt.ylabel('Average Fitness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n\n--- Overall Comparison Summary ---")
    for result in results_summary:
        print(f"Algorithm: {result['name']}")
        print(f"  Best Fitness: {result['best_fitness']:.4f}")
        print(f"  Total Time:   {result['total_time']:.2f}s")
        if result['fitness_history']: 
             print(f"  Best Fitness at last iteration: {result['fitness_history'][-1]:.4f}")
        if result['avg_fitness_history']: # Check if avg_fitness_history is not empty
            if result['avg_fitness_history'][-1] != -np.inf : # Check for valid avg fitness
                print(f"  Avg Fitness at last iteration: {result['avg_fitness_history'][-1]:.4f}")
            else:
                print(f"  Avg Fitness at last iteration: N/A")

        print("-" * 30)

    print("\nTo see detailed iteration logs for each algorithm, set DEBUG_MODE = True in utils/debugging.py")

