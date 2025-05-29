


import numpy as np
from feda_algorithm.optimizer import RF_MIMIC, MIMIC_O2 # Import both optimizers
from problem_definitions.nk_landscape import NKLandscapeProblem
from utils.debugging import print_debug, DEBUG_MODE

# To see debug messages from the algorithm, set DEBUG_MODE to True in utils/debugging.py
# import utils.debugging
# utils.debugging.DEBUG_MODE = True # Uncomment to enable debug prints

if __name__ == "__main__":
    print("Comparing RF_MIMIC and MIMIC_O2 on NK-Landscape Problem...")

    # 1. Define the problem (once for both algorithms)
    N_genes = 50
    K_interactions = 10
    landscape_seed = 50 # Ensure the same landscape for both
    problem = NKLandscapeProblem(N=N_genes, K=K_interactions, landscape_seed=landscape_seed)
    print(f"Problem: {problem} with landscape_seed={landscape_seed}")

    # Define algorithm configurations
    # Note: We standardized to optimizer.population_size and optimizer.elite_ratio in optimizer.py
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
                "elite_ratio": 0.2 # Standardized to elite_ratio
            }
        }
    ]

    results_summary = []
    run_random_seed = 123 # Use the same seed for the optimization run itself for fair comparison

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
        best_solution, best_fitness, fitness_history, timing_info = optimizer.run(random_seed=run_random_seed)

        print(f"\n--- {algo_name} Optimization Complete ---")
        total_time = timing_info.get('total_time', -1.0)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Best fitness achieved: {best_fitness:.4f}")
        if best_solution is not None:
            # print(f"Best solution found: {best_solution}") # Can be long for N=50
            print(f"Best solution found (first 10 genes): {best_solution[:10]}...")
        else:
            print("No solution found.")
        
        results_summary.append({
            "name": algo_name,
            "best_fitness": best_fitness,
            "total_time": total_time,
            "fitness_history": fitness_history # For potential plotting later
        })

    # 4. Display comparative results (simple version)
    print("\n\n--- Overall Comparison Summary ---")
    for result in results_summary:
        print(f"Algorithm: {result['name']}")
        print(f"  Best Fitness: {result['best_fitness']:.4f}")
        print(f"  Total Time:   {result['total_time']:.2f}s")
        if result['fitness_history']:
             print(f"  Fitness at last iteration: {result['fitness_history'][-1]:.4f}")
        print("-" * 30)

    print("\nTo see detailed iteration logs for each algorithm, set DEBUG_MODE = True in utils/debugging.py")
    print("The debug logs will be interleaved if DEBUG_MODE is True for both runs.")

