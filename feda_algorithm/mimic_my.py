# feda_project/feda_algorithm/mimic_my.py
"""
Implements a simpler MIMIC variant using only marginal probabilities.
"""

import numpy as np
import time
import random # Ensure random is imported if np.random.rand() is used for decisions
from utils.debugging import print_debug

class MIMIC_MY:
    """
    A simpler MIMIC variant using only marginal probabilities (no dependency tree).
    Standardized attributes: self.population_size, self.elite_ratio, self.num_genes
    """
    def __init__(self, problem, population_size=500, elite_ratio=0.2, max_iterations=100): # Changed elite_fraction to elite_ratio
        self.problem = problem
        self.population_size = population_size
        self.elite_ratio = elite_ratio # Standardized: was elite_fraction
        self.max_iterations = max_iterations
        
        if hasattr(problem, 'num_genes'): # Prioritize num_genes
            self.num_genes = problem.num_genes
        elif hasattr(problem, 'num_items'):
            self.num_genes = problem.num_items 
        else:
            raise ValueError("Problem object must have 'num_items' or 'num_genes' attribute.")

        self.timing_info = {'total_time': 0.0, 'iteration_times': []}
        self.random_seed = None
        self.fitness_history = [] # Tracks best fitness per iteration
        self.avg_fitness_history = [] # ADDED: Tracks average fitness per iteration
        self._current_fitness_scores = np.array([]) # Internal: to store fitness of current population

    def estimate_probabilities(self, elite: np.ndarray) -> np.ndarray:
        """Estimates marginal probabilities from elite samples."""
        if self.num_genes == 0:
            return np.array([]) 
        if elite.shape[0] == 0 : 
            print_debug("[MIMIC_MY] Warning: No elite samples to estimate probabilities. Defaulting to uniform.")
            return np.full(self.num_genes, 0.5) 
        return np.mean(elite, axis=0)

    def sample_from_distribution(self, prob_model: np.ndarray) -> np.ndarray:
        """Samples a new population based on the marginal probability model."""
        if self.population_size == 0:
            return np.array([]).reshape(0, self.num_genes) 
        if self.num_genes == 0:
            return np.array([]).reshape(self.population_size, 0) 

        if prob_model.size == 0 and self.num_genes > 0 :
            print_debug("[MIMIC_MY] Warning: prob_model empty for non-zero genes. Defaulting to uniform.")
            prob_model = np.full(self.num_genes, 0.5)
        elif prob_model.size != self.num_genes and self.num_genes > 0: 
            print_debug(f"[MIMIC_MY] Warning: prob_model size {prob_model.size} != num_genes {self.num_genes}. Defaulting to uniform.")
            prob_model = np.full(self.num_genes, 0.5)

        return (np.random.rand(self.population_size, self.num_genes) < prob_model).astype(int)

    def run(self, random_seed=None):
        if random_seed is not None:
            self.random_seed = random_seed
            np.random.seed(self.random_seed)
            random.seed(self.random_seed) 
        elif self.random_seed is not None: 
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        overall_start_time = time.time()
        self.timing_info = {'total_time': 0.0, 'iteration_times': []}
        self.fitness_history = [] 
        self.avg_fitness_history = [] 

        init_start_time = time.time()

        if self.population_size == 0 :
            population = np.array([]).reshape(0, self.num_genes)
        elif self.num_genes == 0 :
            population = np.array([]).reshape(self.population_size, 0)
        else:
            population = np.random.randint(2, size=(self.population_size, self.num_genes))

        overall_best_solution = None
        overall_best_fitness = -np.inf
        initial_avg_fitness_val = -np.inf

        if self.num_genes == 0: 
            overall_best_solution = np.array([]).reshape(0,)
            overall_best_fitness = self.problem.fitness(overall_best_solution) 
            initial_avg_fitness_val = overall_best_fitness
            self._current_fitness_scores = np.full(self.population_size, overall_best_fitness) if self.population_size > 0 else np.array([])
        elif self.population_size == 0: 
            overall_best_solution = None 
            overall_best_fitness = -np.inf
            self._current_fitness_scores = np.array([])
        else: 
            if population.shape[0] > 0: 
                self._current_fitness_scores = np.array([self.problem.fitness(ind) for ind in population])
                if self._current_fitness_scores.size > 0:
                    max_fitness_idx = np.argmax(self._current_fitness_scores)
                    overall_best_fitness = self._current_fitness_scores[max_fitness_idx]
                    overall_best_solution = population[max_fitness_idx].copy()
                    initial_avg_fitness_val = np.mean(self._current_fitness_scores)
            else: 
                self._current_fitness_scores = np.array([])


        self.fitness_history.append(overall_best_fitness)
        self.avg_fitness_history.append(initial_avg_fitness_val) 
        self.timing_info['iteration_times'].append(time.time() - init_start_time)

        print_debug(f"[MIMIC_MY] Initial Best Fitness: {overall_best_fitness:.4f}")
        if self._current_fitness_scores.size > 0:
             print_debug(f"[MIMIC_MY] Initial Avg Fitness: {initial_avg_fitness_val:.4f}")


        if self.population_size == 0: 
            self.timing_info['total_time'] = time.time() - overall_start_time
            print_debug(f"[MIMIC_MY] Optimization finished (pop_size=0). Total time: {self.timing_info['total_time']:.2f}s")
            return overall_best_solution, overall_best_fitness, self.fitness_history, self.avg_fitness_history, self.timing_info

        elite_size = max(1, int(self.population_size * self.elite_ratio)) 
        elite_size = min(elite_size, self.population_size) 

        for iteration in range(self.max_iterations):
            iter_start_time = time.time()
            
            current_iter_pop_best_fitness = -np.inf 
            current_iter_pop_avg_fitness = -np.inf

            if self._current_fitness_scores.size > 0:
                current_max_idx = np.argmax(self._current_fitness_scores)
                current_iter_pop_best_fitness = self._current_fitness_scores[current_max_idx]
                current_iter_pop_avg_fitness = np.mean(self._current_fitness_scores) 

                if current_iter_pop_best_fitness > overall_best_fitness:
                    overall_best_fitness = current_iter_pop_best_fitness
                    if population.shape[0] > 0 and current_max_idx < population.shape[0]: 
                        overall_best_solution = population[current_max_idx].copy()
            elif self.num_genes == 0: 
                current_iter_pop_best_fitness = overall_best_fitness
                current_iter_pop_avg_fitness = overall_best_fitness


            self.fitness_history.append(overall_best_fitness) 
            self.avg_fitness_history.append(current_iter_pop_avg_fitness) 

            if self.num_genes > 0 and population.shape[0] > 0 and \
               self._current_fitness_scores.size > 0 and \
               elite_size > 0 and elite_size <= population.shape[0] :
                elite_indices = np.argsort(self._current_fitness_scores)[-elite_size:]
                elite = population[elite_indices]
                if elite.shape[0] > 0 :
                    prob_model = self.estimate_probabilities(elite)
                    population = self.sample_from_distribution(prob_model)
                else: 
                    print_debug(f"[MIMIC_MY] Iter {iteration+1}: No elites selected despite conditions. Re-initializing.")
                    population = np.random.randint(2, size=(self.population_size, self.num_genes))
            elif self.num_genes > 0 and self.population_size > 0 : 
                print_debug(f"[MIMIC_MY] Iter {iteration+1}: Conditions for elite selection not met. Re-initializing population.")
                population = np.random.randint(2, size=(self.population_size, self.num_genes))
            
            if self.population_size > 0 and self.num_genes > 0:
                 self._current_fitness_scores = np.array([self.problem.fitness(ind) for ind in population])
            elif self.num_genes == 0 and self.population_size > 0 :
                 self._current_fitness_scores = np.full(self.population_size, overall_best_fitness)


            iter_time_taken = time.time() - iter_start_time
            self.timing_info['iteration_times'].append(iter_time_taken)

            if (iteration + 1) % 10 == 0 or iteration == self.max_iterations - 1:
                print_debug(f"[MIMIC_MY] Iteration {iteration+1}/{self.max_iterations}: "
                            f"Iter Pop Best (before select) = {current_iter_pop_best_fitness:.4f}, "
                            f"Iter Pop Avg (before select) = {current_iter_pop_avg_fitness:.4f}, "
                            f"Overall Best = {overall_best_fitness:.4f}, Time: {iter_time_taken:.2f}s")

        self.timing_info['total_time'] = time.time() - overall_start_time
        print_debug(f"[MIMIC_MY] Optimization finished. Total time: {self.timing_info['total_time']:.2f}s")
        print_debug(f"[MIMIC_MY] Final Best Fitness: {overall_best_fitness:.4f}")
        if self._current_fitness_scores.size > 0:
            final_avg_fitness = np.mean(self._current_fitness_scores)
            print_debug(f"[MIMIC_MY] Final Avg Fitness of last population: {final_avg_fitness:.4f}")

        return overall_best_solution, overall_best_fitness, self.fitness_history, self.avg_fitness_history, self.timing_info
