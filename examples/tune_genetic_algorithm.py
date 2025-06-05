# feda_project/examples/tune_genetic_algorithm.py
import sys
import os
import time # For overall script timing if needed

# Ensure the project root is in the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from feda_algorithm.genetic_algorithm import GeneticAlgorithm
from problem_definitions.nk_landscape import NKLandscapeProblem
from utils.debugging import print_debug, DEBUG_MODE

# To see detailed iteration logs from the algorithm, set DEBUG_MODE to True in utils/debugging.py
# import utils.debugging
# utils.debugging.DEBUG_MODE = True # Uncomment to enable detailed GA logs

if __name__ == "__main__":
    print("Fine-tuning Genetic Algorithm for NK-Landscape Problem (N=50, K=20)...")

    # 1. Define the problem
    N_genes = 50
    K_interactions = 20
    landscape_seed = 40  # Consistent landscape for all tuning runs
    problem = NKLandscapeProblem(N=N_genes, K=K_interactions, landscape_seed=landscape_seed)
    print(f"Problem: {problem} with landscape_seed={landscape_seed}")

    # 2. Define hyperparameter configurations to test for GeneticAlgorithm
    # You can expand this list with more combinations
    ga_configs_to_test = [
        {
            "id": "GA_Config_1_Base",
            "params": {
                "population_size": 1000, "max_iterations": 50, "elite_ratio": 0.1,
                "crossover_rate": 0.9, "mutation_rate": 0.01, "tournament_size": 3
            }
        },
        {
            "id": "GA_Config_2_LargerPop",
            "params": {
                "population_size": 2000, "max_iterations": 50, "elite_ratio": 0.1,
                "crossover_rate": 0.9, "mutation_rate": 0.01, "tournament_size": 3
            }
        },
        {
            "id": "GA_Config_2_LargerPop",
            "params": {
                "population_size": 3000, "max_iterations": 50, "elite_ratio": 0.1,
                "crossover_rate": 0.9, "mutation_rate": 0.01, "tournament_size": 3
            }
        }
        
        
    ]

    tuning_results = []
    # Use a consistent seed for the GA's internal randomness across different param sets
    # to make comparisons of parameter effects more direct for a single problem instance.
    # For true hyperparameter optimization, you might average over several run_random_seeds.
    ga_run_seed = 123 

    print(f"\nRunning GA with {len(ga_configs_to_test)} different configurations (run_seed={ga_run_seed}):")
    print("-" * 70)

    # Prepare for plotting average fitness histories
    plt.figure(figsize=(14, 9))
    
    overall_script_start_time = time.time()

    for config_info in ga_configs_to_test:
        config_id = config_info["id"]
        params = config_info["params"]

        print(f"\n--- Testing Configuration: {config_id} ---")
        
        # Add the problem to the params for the optimizer
        current_config = {"problem": problem, **params}
        
        optimizer = GeneticAlgorithm(**current_config)
        
        print(f"Parameters: pop_size={optimizer.population_size}, max_iter={optimizer.max_iterations}, elite_ratio={optimizer.elite_ratio}, "
              f"cx_rate={optimizer.crossover_rate}, mut_rate={optimizer.mutation_rate}, tourney_size={optimizer.tournament_size}")

        print(f"Starting optimization for {config_id}...")
        best_solution, best_fitness, fitness_history, avg_fitness_history, timing_info = optimizer.run(random_seed=ga_run_seed)
        
        total_time = timing_info.get('total_time', -1.0)
        print(f"--- {config_id} Complete ---")
        print(f"  Best Fitness Achieved: {best_fitness:.4f}")
        print(f"  Total Execution Time: {total_time:.2f} seconds")
        if avg_fitness_history and avg_fitness_history[-1] != -np.inf:
            print(f"  Avg Fitness at last iteration: {avg_fitness_history[-1]:.4f}")
        else:
            print(f"  Avg Fitness at last iteration: N/A")


        tuning_results.append({
            "id": config_id,
            "params": params,
            "best_fitness": best_fitness,
            "total_time": total_time,
            "avg_fitness_history": avg_fitness_history
        })

        # Plot average fitness for this configuration
        if avg_fitness_history:
            iterations = range(len(avg_fitness_history))
            plt.plot(iterations, avg_fitness_history, label=f'{config_id} (Best: {best_fitness:.4f})')

    overall_script_end_time = time.time()
    print(f"\nTotal tuning script execution time: {overall_script_end_time - overall_script_start_time:.2f} seconds")

    # Configure and display the plot
    plt.title(f'GA Average Population Fitness vs. Iteration for Different Configs (N={N_genes}, K={K_interactions})')
    plt.xlabel('Iteration (0 = Initial Population)')
    plt.ylabel('Average Fitness')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print a summary of tuning results
    print("\n\n--- Hyperparameter Tuning Summary ---")
    # Sort results by best fitness (descending)
    sorted_results = sorted(tuning_results, key=lambda x: x["best_fitness"], reverse=True)
    for result in sorted_results:
        print(f"Config ID: {result['id']}")
        print(f"  Best Fitness: {result['best_fitness']:.4f}")
        print(f"  Total Time:   {result['total_time']:.2f}s")
        if result['avg_fitness_history'] and result['avg_fitness_history'][-1] != -np.inf:
             print(f"  Final Avg Fitness: {result['avg_fitness_history'][-1]:.4f}")
        print(f"  Params: {result['params']}")
        print("-" * 40)

    print("\nNote: For robust hyperparameter tuning, consider running each configuration multiple times with different 'ga_run_seed' values and averaging the results.")

