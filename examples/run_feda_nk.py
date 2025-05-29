# feda_project/examples/run_feda_nk.py
import sys
import os
# Ensure the project root is in the Python path
# For the provided structure:
# sys.path.append("D:/New_folder/Python_Project/feda-project") # User's specific path
# A more relative path might be (uncomment and adjust if needed):
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) 
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


import numpy as np
import matplotlib.pyplot as plt # Import matplotlib
from feda_algorithm.optimizer import RF_MIMIC, MIMIC_O2 
from problem_definitions.nk_landscape import NKLandscapeProblem
from utils.debugging import print_debug, DEBUG_MODE

# To see debug messages from the algorithm, set DEBUG_MODE to True in utils/debugging.py
# import utils.debugging
# utils.debugging.DEBUG_MODE = True # Uncomment to enable debug prints

if __name__ == "__main__":
    print("Comparing RF_MIMIC and MIMIC_O2 on NK-Landscape Problem...")

    # 1. Define the problem (once for both algorithms)
    N_genes = 50
    K_interactions = 20
    landscape_seed = 40 
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
                "population_size": 1000,
                "max_iterations": 50,
                "elite_ratio": 0.2 
            }
        }
    ]

    results_summary = []
    run_random_seed = 123 

    plt.figure(figsize=(12, 8)) # Create a figure for the plot

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
        # MODIFIED: Unpack avg_fitness_history
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
            "avg_fitness_history": avg_fitness_history # Store for plotting
        })

        # Plot average fitness for this algorithm
        if avg_fitness_history:
            iterations = range(len(avg_fitness_history)) # Should be max_iterations + 1 (for initial)
            plt.plot(iterations, avg_fitness_history, label=f'{algo_name} Avg Fitness')

    # Configure and display the plot
    plt.title(f'Average Population Fitness vs. Iteration (N={N_genes}, K={K_interactions})')
    plt.xlabel('Iteration')
    plt.ylabel('Average Fitness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Display comparative results (simple version)
    print("\n\n--- Overall Comparison Summary ---")
    for result in results_summary:
        print(f"Algorithm: {result['name']}")
        print(f"  Best Fitness: {result['best_fitness']:.4f}")
        print(f"  Total Time:   {result['total_time']:.2f}s")
        if result['fitness_history']: # This is best fitness history
             print(f"  Best Fitness at last iteration: {result['fitness_history'][-1]:.4f}")
        if result['avg_fitness_history']:
            print(f"  Avg Fitness at last iteration: {result['avg_fitness_history'][-1]:.4f}")

        print("-" * 30)

    print("\nTo see detailed iteration logs for each algorithm, set DEBUG_MODE = True in utils/debugging.py")

