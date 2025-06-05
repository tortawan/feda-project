# feda_project/feda_algorithm/genetic_algorithm.py
"""
Implements a standard Genetic Algorithm (GA).
"""

import numpy as np
import time
import random
from utils.debugging import print_debug

class GeneticAlgorithm:
    """
    Implements a standard Genetic Algorithm (GA) for binary string optimization.
    Operators: Tournament selection, one-point crossover, bit-flip mutation, elitism.
    """
    def __init__(self, problem, population_size: int = 100,
                 max_iterations: int = 100, elite_ratio: float = 0.1, # Using elite_ratio for consistency
                 crossover_rate: float = 0.9, mutation_rate: float = 0.01,
                 tournament_size: int = 3):
        
        if hasattr(problem, 'num_genes'):
            self.num_genes = problem.num_genes
        elif hasattr(problem, 'num_items'):
            self.num_genes = problem.num_items
        else:
            raise ValueError("The 'problem' object must have a 'num_genes' or 'num_items' attribute.")

        if hasattr(problem, 'fitness') and callable(problem.fitness):
            self.fitness_fn = problem.fitness
        else:
            raise ValueError("The 'problem' object must have a callable 'fitness' method.")

        self.population_size = population_size
        self.max_iterations = max_iterations
        self.elite_ratio = elite_ratio # Percentage of population to carry over as elites
        self.elitism_count = int(self.population_size * self.elite_ratio) if self.population_size > 0 else 0
        if self.population_size > 0 and self.elitism_count == 0 and self.elite_ratio > 0: # Ensure at least 1 elite if ratio > 0 and pop > 0
            self.elitism_count = 1


        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate # Per-gene mutation rate
        self.tournament_size = tournament_size
        
        self.random_seed = None
        self.population = None
        self.fitnesses = None
        self.best_solution = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.avg_fitness_history = []
        self.timing_info = {'total_time': 0.0, 'iteration_times': []}

    def _initialize_population(self) -> np.ndarray:
        if self.num_genes == 0:
            return np.array([]).reshape(self.population_size, 0)
        return np.random.randint(0, 2, size=(self.population_size, self.num_genes), dtype=int)

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        if population.shape[0] == 0:
            return np.array([])
        return np.array([self.fitness_fn(ind) for ind in population])

    def _selection(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """Performs tournament selection."""
        selected_parents = []
        for _ in range(population.shape[0]): # Select N parents to create N offspring
            tournament_indices = np.random.choice(population.shape[0], self.tournament_size, replace=True)
            tournament_fitnesses = fitnesses[tournament_indices]
            winner_idx_in_tournament = np.argmax(tournament_fitnesses)
            selected_parents.append(population[tournament_indices[winner_idx_in_tournament]])
        return np.array(selected_parents)

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """Performs one-point crossover."""
        if self.num_genes == 0:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()
        if random.random() < self.crossover_rate and self.num_genes > 0:
            point = random.randint(1, self.num_genes - 1) if self.num_genes > 1 else 0
            if point > 0 : # Ensure crossover happens only if point allows for swapping
                child1[point:], child2[point:] = parent2[point:], parent1[point:]
        return child1, child2

    def _mutation(self, individual: np.ndarray) -> np.ndarray:
        """Performs bit-flip mutation."""
        mutated_individual = individual.copy()
        for i in range(self.num_genes):
            if random.random() < self.mutation_rate:
                mutated_individual[i] = 1 - mutated_individual[i] # Flip bit
        return mutated_individual

    def run(self, random_seed: int = None):
        if random_seed is not None:
            self.random_seed = random_seed
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
        elif self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
        else: # First run and no seed provided
            self.random_seed = np.random.randint(0, 1_000_000)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)


        overall_start_time = time.time()
        self.fitness_history = []
        self.avg_fitness_history = []
        self.timing_info = {'total_time': 0.0, 'iteration_times': []}
        self.best_solution = None
        self.best_fitness = -np.inf

        if self.population_size == 0:
            print_debug("[GeneticAlgorithm] Population size is 0. Returning.")
            self.timing_info['total_time'] = time.time() - overall_start_time
            if self.num_genes == 0:
                empty_sol_fitness = self.fitness_fn(np.array([]).reshape(0,))
                return np.array([]).reshape(0,), empty_sol_fitness, [], [empty_sol_fitness], self.timing_info
            else:
                return None, -np.inf, [], [], self.timing_info

        init_time_start = time.time()
        self.population = self._initialize_population()
        self.fitnesses = self._evaluate_population(self.population)

        current_avg_fitness_for_iteration = -np.inf
        if self.fitnesses.size > 0:
            current_avg_fitness_for_iteration = np.mean(self.fitnesses)
            current_best_idx_iter = np.argmax(self.fitnesses)
            self.best_fitness = self.fitnesses[current_best_idx_iter]
            self.best_solution = self.population[current_best_idx_iter].copy()
        elif self.num_genes == 0: # 0-gene problem
             self.best_fitness = self.fitness_fn(np.array([]).reshape(0,))
             self.best_solution = np.array([]).reshape(0,)
             current_avg_fitness_for_iteration = self.best_fitness


        self.fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(current_avg_fitness_for_iteration)
        self.timing_info['iteration_times'].append(time.time() - init_time_start)

        print_debug(f"[GeneticAlgorithm] Initial Best Fitness: {self.best_fitness:.4f}")
        if self.fitnesses.size > 0 or (self.num_genes == 0 and self.population_size > 0):
            print_debug(f"[GeneticAlgorithm] Initial Avg Fitness: {current_avg_fitness_for_iteration:.4f}")


        for iteration in range(self.max_iterations):
            iter_start_time = time.time()

            new_population = []

            # Elitism: Carry over the best individuals
            if self.elitism_count > 0 and self.fitnesses.size > 0:
                elite_indices = np.argsort(self.fitnesses)[-self.elitism_count:]
                new_population.extend(self.population[elite_indices])

            # Generate the rest of the population
            num_offspring_needed = self.population_size - len(new_population)
            
            if num_offspring_needed > 0 and self.population.shape[0] > 0 : # Check if population exists for selection
                selected_parents = self._selection(self.population, self.fitnesses)
                
                for i in range(0, num_offspring_needed, 2): # Create offspring in pairs
                    if i + 1 < selected_parents.shape[0]: # Ensure there's a pair
                        parent1 = selected_parents[i]
                        parent2 = selected_parents[i+1]
                        child1, child2 = self._crossover(parent1, parent2)
                        new_population.append(self._mutation(child1))
                        if len(new_population) < self.population_size:
                            new_population.append(self._mutation(child2))
                    elif i < selected_parents.shape[0] : # Odd number, last parent mutated
                         new_population.append(self._mutation(selected_parents[i]))
                    # Ensure population size is not exceeded
                    if len(new_population) >= self.population_size:
                        break
            elif num_offspring_needed > 0 : # Population was empty or too small, fill randomly
                 while len(new_population) < self.population_size:
                     new_population.append(self._initialize_population()[0]) # Take one random individual


            self.population = np.array(new_population[:self.population_size]) # Ensure correct size
            self.fitnesses = self._evaluate_population(self.population)

            current_iter_best_fitness = -np.inf
            current_iter_avg_fitness = -np.inf
            if self.fitnesses.size > 0:
                iter_best_idx = np.argmax(self.fitnesses)
                current_iter_best_fitness = self.fitnesses[iter_best_idx]
                current_iter_avg_fitness = np.mean(self.fitnesses)
                if current_iter_best_fitness > self.best_fitness:
                    self.best_fitness = current_iter_best_fitness
                    self.best_solution = self.population[iter_best_idx].copy()
            elif self.num_genes == 0: # 0-gene problem
                current_iter_best_fitness = self.best_fitness
                current_iter_avg_fitness = self.best_fitness
                
            self.fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(current_iter_avg_fitness)
            iter_time_taken = time.time() - iter_start_time
            self.timing_info['iteration_times'].append(iter_time_taken)

            if (iteration + 1) % 10 == 0 or iteration == self.max_iterations - 1:
                print_debug(f"[GeneticAlgorithm] Iteration {iteration + 1}/{self.max_iterations}: "
                            f"Iter Pop Best = {current_iter_best_fitness:.4f}, "
                            f"Iter Pop Avg = {current_iter_avg_fitness:.4f}, "
                            f"Overall Best = {self.best_fitness:.4f}, Time: {iter_time_taken:.2f}s")

        self.timing_info['total_time'] = time.time() - overall_start_time
        print_debug(f"[GeneticAlgorithm] Optimization finished. Total time: {self.timing_info['total_time']:.2f}s")
        print_debug(f"[GeneticAlgorithm] Final Best Fitness: {self.best_fitness:.4f}")
        if self.fitnesses is not None and self.fitnesses.size > 0:
            final_avg_fitness = np.mean(self.fitnesses)
            print_debug(f"[GeneticAlgorithm] Final Avg Fitness of last population: {final_avg_fitness:.4f}")
        
        return self.best_solution, self.best_fitness, self.fitness_history, self.avg_fitness_history, self.timing_info

