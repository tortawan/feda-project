# feda_project/examples/run_feda_nk.py
import numpy as np
from feda_algorithm.optimizer import RF_MIMIC # Or your chosen class name
from problem_definitions.nk_landscape import NKLandscapeProblem
from utils.debugging import print_debug, DEBUG_MODE

# To see debug messages from the algorithm
# import utils.debugging
# utils.debugging.DEBUG_MODE = True

if __name__ == "__main__":
    print("Running FEDA on NK-Landscape Problem...")

    # 1. Define the problem
    N_genes = 20
    K_interactions = 3
    problem = NKLandscapeProblem(N=N_genes, K=K_interactions, landscape_seed=42)
    print(f"Problem: {problem}")

    # 2. Configure the FEDA (RF_MIMIC) optimizer
    # Note: Using the class name RF_MIMIC for the implementation
    # of the Forest-guided Estimation of Distributions Algorithm (FEDA)
    feda_optimizer = RF_MIMIC(
        problem=problem,
        population_size=100,
        max_iterations=50, # Keep low for a quick example
        elite_ratio=0.25,
        rf_params={'n_estimators': 25, 'min_samples_leaf': 2, 'max_depth': 10},
        branch_alpha=0.05
    )
    print(f"Optimizer configured with pop_size={feda_optimizer.pop_size}, max_iter={feda_optimizer.max_iterations}")

    # 3. Run the optimization
    print("\nStarting optimization...")
    best_solution, best_fitness, fitness_history, timing_info = feda_optimizer.run(random_seed=123)

    # 4. Display results
    print("\n--- Optimization Complete ---")
    print(f"Total execution time: {timing_info['total_time']:.2f} seconds")
    print(f"Best fitness achieved: {best_fitness:.4f}")
    if best_solution is not None:
        print(f"Best solution found: {best_solution}")
    else:
        print("No solution found (this might happen with pop_size=0).")

    # print("\nFitness history (best fitness per iteration):")
    # for i, fit_val in enumerate(fitness_history):
    #     print(f"Iteration {i}: {fit_val:.4f}")

    # You could add plotting for fitness_history here if matplotlib is a dependency
    print("\nTo see detailed iteration logs, set DEBUG_MODE = True in utils/debugging.py")