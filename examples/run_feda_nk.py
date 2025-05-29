# feda_project/examples/run_feda_nk.py
import sys
import os
# Ensure the project root is in the Python path
# Adjust this path if your script is located elsewhere relative to the project root
# or if your project structure is different.
# Example: If feda-project is the root and this script is in feda-project/examples
# then ".." should go up one level to feda-project.
# If your current working directory *is* feda-project, you might not need this.
# For the provided structure: sys.path.append("D:/New_folder/Python_Project/feda-project")
# A more relative path might be:
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) # Goes up one level from 'examples'
# sys.path.append(PROJECT_ROOT)


import numpy as np
from feda_algorithm.optimizer import RF_MIMIC, MIMIC_O2 # Import both optimizers
from problem_definitions.nk_landscape import NKLandscapeProblem
from utils.debugging import print_debug, DEBUG_MODE

# To see debug messages from the algorithm, set DEBUG_MODE to True in utils/debugging.py
# import utils.debugging
# utils.debugging.DEBUG_MODE = True # Uncomment to enable debug prints

if __name__ == "__main__":
    # --- CHOOSE ALGORITHM TO RUN ---
    # Options: "RF_MIMIC", "MIMIC_O2"
    ALGORITHM_TO_RUN = "RF_MIMIC"
    #ALGORITHM_TO_RUN = "MIMIC_O2" # Uncomment to run MIMIC_O2
    # -------------------------------

    print(f"Running {ALGORITHM_TO_RUN} on NK-Landscape Problem...")

    # 1. Define the problem
    N_genes = 50
    K_interactions = 10
    problem = NKLandscapeProblem(N=N_genes, K=K_interactions, landscape_seed=42)
    print(f"Problem: {problem}")

    # 2. Configure the chosen optimizer
    optimizer = None
    if ALGORITHM_TO_RUN == "RF_MIMIC":
        print("Configuring FEDA (RF_MIMIC) optimizer...")
        # Note: Using the class name RF_MIMIC for the implementation
        # of the Forest-guided Estimation of Distributions Algorithm (FEDA)
        optimizer = RF_MIMIC(
            problem=problem,
            population_size=3000,
            max_iterations=50, # Keep low for a quick example
            elite_ratio=0.25,  # RF_MIMIC uses 'elite_ratio'
            rf_params={'n_estimators': 200, 'min_samples_leaf': 2, 'max_depth': 25},
            branch_alpha=0.05
        )
    elif ALGORITHM_TO_RUN == "MIMIC_O2":
        print("Configuring MIMIC_O2 optimizer...")
        optimizer = MIMIC_O2(
            problem=problem,
            population_size=1000, # MIMIC_O2 default is 500, using 100 for quicker example
            max_iterations=50,
            elite_ratio=0.2  # MIMIC_O2 uses 'elite_percent'
        )
    else:
        raise ValueError(f"Unknown algorithm specified: {ALGORITHM_TO_RUN}. Choose 'RF_MIMIC' or 'MIMIC_O2'.")

    print(f"Optimizer configured with pop_size={optimizer.population_size}, max_iter={optimizer.max_iterations}")

    # 3. Run the optimization
    print("\nStarting optimization...")
    # Both optimizers should have a 'run' method with a similar signature
    best_solution, best_fitness, fitness_history, timing_info = optimizer.run(random_seed=123)

    # 4. Display results
    print(f"\n--- {ALGORITHM_TO_RUN} Optimization Complete ---")
    if timing_info and 'total_time' in timing_info:
        print(f"Total execution time: {timing_info['total_time']:.2f} seconds")
    else:
        print("Timing information not available or incomplete.")

    print(f"Best fitness achieved: {best_fitness:.4f}")
    if best_solution is not None:
        print(f"Best solution found: {best_solution}")
    else:
        print("No solution found (this might happen with pop_size=0 or other issues).")

    # print("\nFitness history (best fitness per iteration):")
    # if fitness_history:
    #     for i, fit_val in enumerate(fitness_history):
    #         print(f"Iteration {i}: {fit_val:.4f}")
    # else:
    #     print("Fitness history not available.")

    # You could add plotting for fitness_history here if matplotlib is a dependency
    # and you modify the script to collect and plot data for both algorithms if run sequentially.
    print("\nTo see detailed iteration logs, set DEBUG_MODE = True in utils/debugging.py")
    print(f"Make sure utils.debugging.DEBUG_MODE is set to True to see '{ALGORITHM_TO_RUN}' specific logs.")

