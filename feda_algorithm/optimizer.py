# feda_project/feda_algorithm/optimizer.py
"""
Implements the core Forest-guided Estimation of Distributions Algorithm (FEDA)
and a MIMIC (Mutual Information Maximizing Input Clustering) variant.

This module contains:
- RF_MIMIC class: Uses a Random Forest classifier to guide sampling.
- MIMIC_O2 class: Uses a dependency tree built from mutual information.
"""

import numpy as np
import time
import random
from sklearn.ensemble import RandomForestClassifier
from utils.debugging import print_debug # Assuming utils is a sibling directory
from collections import defaultdict # Added for MIMIC_O2

class RF_MIMIC:
    """
    Implements the Forest-guided Estimation of Distributions Algorithm (FEDA).
    Standardized attributes: self.population_size, self.elite_ratio, self.num_genes
    """
    def __init__(self, problem, population_size: int = 100,
                 max_iterations: int = 100, elite_ratio: float = 0.2,
                 rf_params: dict = None, branch_alpha: float = 0.1):
        """Initializes the RF_MIMIC (FEDA) optimizer.
        """
        if hasattr(problem, 'num_genes'):
            self.num_genes = problem.num_genes
        elif hasattr(problem, 'num_items'):
            self.num_genes = problem.num_items # Use num_items as num_genes
        else:
            raise ValueError("The 'problem' object must have a 'num_genes' or 'num_items' attribute.")

        if hasattr(problem, 'fitness') and callable(problem.fitness):
            self.fitness_fn = problem.fitness
        else:
            raise ValueError("The 'problem' object must have a callable 'fitness' method.")

        self.population_size = population_size # Standardized attribute name
        self.max_iterations = max_iterations
        self.elite_ratio = elite_ratio # Standardized attribute name
        self.rf_params = rf_params if rf_params is not None else {}

        self.rf_params.setdefault('n_estimators', 10)
        self.rf_params.setdefault('min_samples_leaf', 1)

        self.branch_alpha = branch_alpha
        self.random_seed = None 

        self.population = None
        self.fitnesses = None
        self.best_solution = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.timing_info = {'total_time': 0.0, 'iteration_times':[]}
        self.rf_model_for_inspection = None

    def _initialize_population(self) -> np.ndarray:
        """Initializes the population with random binary strings."""
        if self.num_genes == 0:
            return np.array([]).reshape(self.population_size, 0) # Use self.population_size
        return np.random.randint(0, 2, size=(self.population_size, self.num_genes), dtype=int) # Use self.population_size

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluates the fitness of each individual in the given population."""
        if population.shape[0] == 0:
            return np.array([])
        return np.array([self.fitness_fn(ind) for ind in population])

    def _select_elite(self, population: np.ndarray, fitnesses: np.ndarray) -> tuple:
        """Selects elite and non-elite individuals from the population."""
        pop_size_current = population.shape[0]
        if pop_size_current == 0:
            return (np.array([]).reshape(0, self.num_genes), np.array([]),
                    np.array([]).reshape(0, self.num_genes), np.array([]))

        # elite_ratio is already used here, consistent with standardization
        if self.elite_ratio < 1.0: 
            elite_count = int(np.floor(pop_size_current * self.elite_ratio))
        else: 
            elite_count = int(self.elite_ratio)

        elite_count = max(1 if pop_size_current > 0 else 0, elite_count)
        elite_count = min(elite_count, pop_size_current)
        
        if elite_count == 0 and pop_size_current > 0:
            elite_count = 1
            
        if fitnesses.size == 0 and pop_size_current > 0 :
            print_debug("RF_MIMIC Warn: Fitnesses empty in _select_elite despite population. Re-evaluating.")
            fitnesses = self._evaluate_population(population)
            if fitnesses.size == 0:
                return (np.array([]).reshape(0,self.num_genes), np.array([]),
                        np.array([]).reshape(0,self.num_genes), np.array([]))

        sorted_idx = np.argsort(fitnesses)[::-1]
        elite_indices = sorted_idx[:elite_count]
        nonelite_indices = sorted_idx[elite_count:]

        elite_pop = population[elite_indices]
        elite_fitnesses = fitnesses[elite_indices]
        nonelite_pop = population[nonelite_indices]
        nonelite_fitnesses = fitnesses[nonelite_indices]

        return elite_pop, elite_fitnesses, nonelite_pop, nonelite_fitnesses

    def _train_random_forest(self, elite_population: np.ndarray,
                             nonelite_population: np.ndarray) -> RandomForestClassifier:
        """Trains a Random Forest classifier to distinguish elites from non-elites."""
        current_rf_params = self.rf_params.copy()
        current_rf_params['random_state'] = self.random_seed

        if elite_population.shape[0] > 0:
            max_min_leaf = max(1, elite_population.shape[0] // 2 if elite_population.shape[0] > 1 else 1)
            if current_rf_params.get('min_samples_leaf', 1) > max_min_leaf:
                current_rf_params['min_samples_leaf'] = max_min_leaf
        
        rf_model_unfitted = RandomForestClassifier(**current_rf_params)

        if elite_population.shape[0] == 0 or nonelite_population.shape[0] == 0:
            print_debug(f"RF_MIMIC Warning: RF training with insufficient distinct classes. Elites: {elite_population.shape[0]}, Non-elites: {nonelite_population.shape[0]}")
            return rf_model_unfitted

        X_train = np.vstack((elite_population, nonelite_population))
        y_train = np.concatenate((
            np.ones(elite_population.shape[0], dtype=int),
            np.zeros(nonelite_population.shape[0], dtype=int)
        ))

        if len(np.unique(y_train)) < 2:
            print_debug(f"RF_MIMIC Warning: Training RF with only one class. Unique labels: {np.unique(y_train)}")
            return rf_model_unfitted
        
        if X_train.shape[0] == 0:
            print_debug("RF_MIMIC Warning: No data to train Random Forest (X_train is empty).")
            return rf_model_unfitted

        try:
            rf_model_unfitted.fit(X_train, y_train)
            self.rf_model_for_inspection = rf_model_unfitted
        except ValueError as e:
            print_debug(f"RF_MIMIC Warning: RandomForestClassifier fitting error: {e}. Returning unfitted model.")
            return RandomForestClassifier(**current_rf_params)
        return rf_model_unfitted

    def _sample_new_population(self, rf_model: RandomForestClassifier,
                               elite_population: np.ndarray) -> np.ndarray:
        """Generates a new population by sampling from the trained Random Forest."""
        new_population_list =[]

        if self.population_size == 0: return np.array([]).reshape(0, self.num_genes) # Use self.population_size
        if self.num_genes == 0: return np.array([]).reshape(self.population_size, 0) # Use self.population_size

        if not hasattr(rf_model, 'estimators_') or not rf_model.estimators_:
            print_debug("RF_MIMIC Warning: RF model not trained or has no trees. Falling back.")
            if elite_population.shape[0] > 0 and self.population_size > 0: # Use self.population_size
                return self._sample_new_population_from_elites_only(elite_population, target_size=self.population_size) # Use self.population_size
            return self._initialize_population()

        trees = rf_model.estimators_
        n_trees = len(trees)
        if n_trees == 0:
            print_debug("RF_MIMIC Warning: RF model has no trees (estimators_ is empty). Falling back.")
            if elite_population.shape[0] > 0 and self.population_size > 0: # Use self.population_size
                return self._sample_new_population_from_elites_only(elite_population, target_size=self.population_size) # Use self.population_size
            return self._initialize_population()

        samples_per_tree = [self.population_size // n_trees] * n_trees # Use self.population_size
        for i in range(self.population_size % n_trees): samples_per_tree[i] += 1 # Use self.population_size

        elite_leaf_indices_per_tree = []
        if elite_population.shape[0] > 0:
            for tree in trees:
                try:
                    if hasattr(tree, 'tree_') and tree.tree_ is not None:
                        elite_leaf_idx = tree.apply(elite_population)
                        elite_leaf_indices_per_tree.append(elite_leaf_idx)
                    else:
                        elite_leaf_indices_per_tree.append(np.array([], dtype=int))
                except Exception: 
                    elite_leaf_indices_per_tree.append(np.array([], dtype=int))
        else:
            for _ in trees: elite_leaf_indices_per_tree.append(np.array([], dtype=int))

        for t_index, tree in enumerate(trees):
            num_samples_from_this_tree = samples_per_tree[t_index]
            if num_samples_from_this_tree == 0: continue

            try:
                tree_struct = tree.tree_
                if tree_struct is None:
                    print_debug(f"RF_MIMIC Warning: Tree {t_index} has no tree_struct (None). Skipping.")
                    continue
            except AttributeError:
                print_debug(f"RF_MIMIC Warning: Tree {t_index} does not have 'tree_' attribute. Skipping.")
                continue

            leaf_to_elite_indices = {}
            if elite_population.shape[0] > 0 and t_index < len(elite_leaf_indices_per_tree) and elite_leaf_indices_per_tree[t_index].size > 0:
                leaf_idx_for_elites_this_tree = elite_leaf_indices_per_tree[t_index]
                for elite_idx, leaf_node_id in enumerate(leaf_idx_for_elites_this_tree):
                    leaf_to_elite_indices.setdefault(leaf_node_id, []).append(elite_idx)

            for _ in range(num_samples_from_this_tree):
                node = 0 
                path_decisions = {} 

                while tree_struct.feature[node]!= -2: 
                    feature_index = tree_struct.feature[node]
                    left_child = tree_struct.children_left[node]
                    right_child = tree_struct.children_right[node]

                    model_classes = getattr(rf_model, 'classes_', np.array([0, 1])) 
                    class_index1 = -1
                    if 1 in model_classes: class_index1 = list(model_classes).index(1)

                    left_count1, right_count1 = 0.0, 0.0
                    if class_index1!= -1:
                        if left_child!= -1 and tree_struct.value[left_child].ndim >=2 and \
                           class_index1 < tree_struct.value[left_child].shape[1]: 
                            left_count1 = tree_struct.value[left_child][0, class_index1]
                        if right_child!= -1 and tree_struct.value[right_child].ndim >=2 and \
                           class_index1 < tree_struct.value[right_child].shape[1]: 
                            right_count1 = tree_struct.value[right_child][0, class_index1]
                    
                    total1 = left_count1 + right_count1
                    denominator = total1 + 2 * self.branch_alpha 
                    p_left = (left_count1 + self.branch_alpha) / denominator if denominator > 1e-9 else 0.5

                    if random.random() < p_left:
                        path_decisions[feature_index] = 0 
                        node = left_child
                    else:
                        path_decisions[feature_index] = 1
                        node = right_child
                    
                    if node == -1: 
                        print_debug("RF_MIMIC Warning: Traversed to a non-existent child node (-1). Breaking path.")
                        break 
                
                if node == -1: continue 

                individual = np.full(self.num_genes, -1, dtype=int) 
                for feat_idx, feat_val in path_decisions.items():
                    if 0 <= feat_idx < self.num_genes:
                        individual[feat_idx] = feat_val

                elites_in_this_leaf_indices = leaf_to_elite_indices.get(node, []) 
                elites_in_this_leaf_samples = elite_population[elites_in_this_leaf_indices] if len(elites_in_this_leaf_indices) > 0 and elite_population.shape[0] > 0 else np.array([[]]).reshape(0,self.num_genes)


                for j in range(self.num_genes):
                    if individual[j] == -1: 
                        if elites_in_this_leaf_samples.shape[0] > 0:
                            prob_one = np.mean(elites_in_this_leaf_samples[:, j])
                        elif elite_population.shape[0] > 0: 
                            prob_one = np.mean(elite_population[:,j])
                        else: 
                            prob_one = 0.5
                        individual[j] = 1 if random.random() < prob_one else 0
                new_population_list.append(individual)

        if not new_population_list and self.population_size > 0: # Use self.population_size
            print_debug("RF_MIMIC Warning: New population list empty after RF sampling. Falling back.")
            if elite_population.shape[0] > 0:
                return self._sample_new_population_from_elites_only(elite_population, target_size=self.population_size) # Use self.population_size
            return self._initialize_population()

        final_population_array = np.array(new_population_list, dtype=int) if new_population_list else np.array([]).reshape(0, self.num_genes)


        if final_population_array.shape[0] < self.population_size and self.population_size > 0: # Use self.population_size
            num_needed = self.population_size - final_population_array.shape[0] # Use self.population_size
            print_debug(f"RF_MIMIC: Population short by {num_needed} after RF sampling. Filling...")
            fill_samples = np.array([]).reshape(0,self.num_genes)
            if elite_population.shape[0] > 0:
                fill_samples = self._sample_new_population_from_elites_only(elite_population, target_size=num_needed)
            else:
                if self.num_genes > 0:
                    fill_samples = np.random.randint(0,2,size=(num_needed, self.num_genes))
                else:
                    fill_samples = np.array([]).reshape(num_needed, 0)
            
            if final_population_array.shape[0] == 0: final_population_array = fill_samples
            elif fill_samples.shape[0] > 0: final_population_array = np.vstack((final_population_array, fill_samples))

        if final_population_array.shape[0] > self.population_size: # Use self.population_size
            final_population_array = final_population_array[:self.population_size] # Use self.population_size
        
        if self.num_genes == 0 and self.population_size > 0: # Use self.population_size
            return np.array([]).reshape(self.population_size, 0) # Use self.population_size
        if self.population_size == 0: # Use self.population_size
            return np.array([]).reshape(0, self.num_genes)
            
        return final_population_array

    def _sample_new_population_from_elites_only(self, elite_population: np.ndarray, target_size: int) -> np.ndarray:
        """Generates new individuals based solely on the elite population's gene frequencies."""
        if target_size == 0: return np.array([]).reshape(0, self.num_genes)
        if self.num_genes == 0: return np.array([]).reshape(target_size, 0)

        new_population = []
        num_elites = elite_population.shape[0]

        if num_elites == 0: 
            return np.random.randint(0, 2, size=(target_size, self.num_genes), dtype=int)

        for _ in range(target_size):
            individual = np.zeros(self.num_genes, dtype=int)
            for j in range(self.num_genes):
                prob_one = np.mean(elite_population[:, j])
                individual[j] = 1 if random.random() < prob_one else 0
            new_population.append(individual)
        return np.array(new_population, dtype=int)


    def run(self, random_seed: int = None):
        """Runs the FEDA optimization process."""
        if random_seed is not None:
            self.random_seed = random_seed
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
        elif self.random_seed is None: 
            self.random_seed = np.random.randint(0, 1_000_000)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        self.rf_params['random_state'] = self.random_seed 

        overall_start_time = time.time()
        self.fitness_history = []
        self.timing_info = {'total_time': 0.0, 'iteration_times':[]}
        self.best_solution = None 
        self.best_fitness = -np.inf 

        if self.population_size == 0: # Use self.population_size
            print_debug("RF_MIMIC: Population size is 0. Returning.")
            self.timing_info['total_time'] = time.time() - overall_start_time
            if self.num_genes == 0:
                empty_sol_fitness = self.fitness_fn(np.array([]).reshape(0,)) 
                return np.array([]).reshape(0,), empty_sol_fitness,[], self.timing_info
            else:
                return None, -np.inf,[], self.timing_info

        init_time_start = time.time()
        self.population = self._initialize_population()
        
        if self.population.shape[0] > 0: 
            self.fitnesses = self._evaluate_population(self.population)
        else: 
            self.fitnesses = np.array([])

        if self.fitnesses is not None and self.fitnesses.size > 0:
            current_best_idx = np.argmax(self.fitnesses)
            self.best_solution = self.population[current_best_idx].copy()
            self.best_fitness = self.fitnesses[current_best_idx]
        else: 
            if self.num_genes == 0 and self.population_size > 0:  # Use self.population_size
                self.best_solution = np.array([]).reshape(0,)
                self.best_fitness = self.fitness_fn(self.best_solution)

        self.fitness_history.append(self.best_fitness)
        self.timing_info['iteration_times'].append(time.time() - init_time_start)
        
        print_debug(f"[RF_MIMIC] Initial Best Fitness: {self.best_fitness:.4f}")
        if self.fitnesses is not None and self.fitnesses.size > 0:
            initial_avg_fitness = np.mean(self.fitnesses)
            print_debug(f"[RF_MIMIC] Initial Avg Fitness: {initial_avg_fitness:.4f}")

        for iteration in range(self.max_iterations):
            iter_start_time = time.time()

            elite_pop, _, nonelite_pop, _ = \
                self._select_elite(self.population, self.fitnesses)
            
            action_taken = "initial_selection_done"

            if elite_pop.shape[0] == 0 and self.population_size > 0: # Use self.population_size
                print_debug(f"FEDA Iter {iteration + 1}: No elite samples. Re-initializing.")
                self.population = self._initialize_population()
                action_taken = "reinit_due_to_no_elites"
            elif nonelite_pop.shape[0] == 0 and elite_pop.shape[0] == self.population_size and self.population_size > 0: # Use self.population_size
                print_debug(f"FEDA Iter {iteration + 1}: All samples are elite. Sampling from elites.")
                self.population = self._sample_new_population_from_elites_only(elite_pop, target_size=self.population_size) # Use self.population_size
                action_taken = "sample_from_all_elites"
            elif self.population_size > 0:  # Use self.population_size
                rf_model = self._train_random_forest(elite_pop, nonelite_pop)
                if hasattr(rf_model, 'estimators_') and rf_model.estimators_ and len(rf_model.estimators_) > 0:
                    self.population = self._sample_new_population(rf_model, elite_pop)
                    action_taken = "sampled_from_trained_rf"
                else:
                    print_debug(f"FEDA Iter {iteration + 1}: RF model not effectively trained. Fallback.")
                    if elite_pop.shape[0] > 0:
                        self.population = self._sample_new_population_from_elites_only(elite_pop, target_size=self.population_size) # Use self.population_size
                        action_taken = "rf_train_fail_fallback_elites"
                    else:
                        self.population = self._initialize_population()
                        action_taken = "rf_train_fail_fallback_random"
            
            if self.population_size > 0 and (self.population.shape[0]!= self.population_size or \
                                      (self.num_genes == 0 and self.population.shape[1]!=0)): # Use self.population_size
                print_debug(f"FEDA Iter {iteration + 1}: Pop size {self.population.shape} not {self.population_size}x{self.num_genes} after {action_taken}. Adjusting.") # Use self.population_size
                if self.population.shape[0] < self.population_size: # Use self.population_size
                    num_needed = self.population_size - self.population.shape[0] # Use self.population_size
                    fill_samples = np.array([]).reshape(0,self.num_genes)
                    if elite_pop.shape[0] > 0 :
                        fill_samples = self._sample_new_population_from_elites_only(elite_pop, target_size=num_needed)
                    else:
                        if self.num_genes > 0:
                            fill_samples = np.random.randint(0, 2, size=(num_needed, self.num_genes), dtype=int)
                        else:
                            fill_samples = np.array([]).reshape(num_needed, 0)
                    
                    if self.population.shape[0] == 0: self.population = fill_samples
                    elif fill_samples.shape[0] > 0: self.population = np.vstack((self.population, fill_samples))
                
                elif self.population.shape[0] > self.population_size: # Use self.population_size
                    self.population = self.population[:self.population_size] # Use self.population_size
                
                if self.num_genes == 0 and self.population.shape!= (self.population_size, 0):  # Use self.population_size
                    self.population = np.array([]).reshape(self.population_size, 0) # Use self.population_size

            if self.population_size > 0 and self.population.shape[0] == 0 and self.num_genes > 0 : # Use self.population_size
                print_debug(f"FEDA Critical Error: Population empty at iter {iteration + 1}. Reinitializing.")
                self.population = self._initialize_population() 

            current_iter_best_fitness = -np.inf
            current_iter_avg_fitness = -np.inf 

            if self.population.shape[0] > 0:
                self.fitnesses = self._evaluate_population(self.population)
                if self.fitnesses.size > 0:
                    current_iter_best_idx = np.argmax(self.fitnesses)
                    current_iter_best_fitness = self.fitnesses[current_iter_best_idx]
                    current_iter_avg_fitness = np.mean(self.fitnesses)

                    if current_iter_best_fitness > self.best_fitness:
                        self.best_fitness = current_iter_best_fitness
                        self.best_solution = self.population[current_iter_best_idx].copy()
            elif self.num_genes == 0 and self.population_size > 0:  # Use self.population_size
                current_iter_best_fitness = self.best_fitness 
                if self.fitnesses is not None and self.fitnesses.size > 0: 
                    current_iter_avg_fitness = np.mean(self.fitnesses) 
                else: 
                    current_iter_avg_fitness = self.best_fitness


            self.fitness_history.append(self.best_fitness)
            iter_time_taken = time.time() - iter_start_time
            self.timing_info['iteration_times'].append(iter_time_taken)

            if (iteration + 1) % 10 == 0 or iteration == self.max_iterations - 1:
                print_debug(f"[RF_MIMIC] Iteration {iteration + 1}/{self.max_iterations}: " # Added Algo Name
                            f"Iter Pop Best = {current_iter_best_fitness:.4f}, "
                            f"Iter Pop Avg = {current_iter_avg_fitness:.4f}, " 
                            f"Overall Best = {self.best_fitness:.4f}, Time: {iter_time_taken:.2f}s")

        self.timing_info['total_time'] = time.time() - overall_start_time
        print_debug(f"[RF_MIMIC] Optimization finished. Total time: {self.timing_info['total_time']:.2f}s") # Added Algo Name
        print_debug(f"[RF_MIMIC] Final Best Fitness: {self.best_fitness:.4f}") # Added Algo Name
        if self.fitnesses is not None and self.fitnesses.size > 0:
            final_avg_fitness = np.mean(self.fitnesses)
            print_debug(f"[RF_MIMIC] Final Avg Fitness of last population: {final_avg_fitness:.4f}") # Added Algo Name
        return self.best_solution, self.best_fitness, self.fitness_history, self.timing_info

# --- MIMIC_O2 Class ---
class MIMIC_O2:
    """
    Implements the MIMIC algorithm using a dependency tree.
    Standardized attributes: self.population_size, self.elite_ratio, self.num_genes
    """
    def __init__(self, problem, population_size=500, elite_ratio=0.2, max_iterations=100): # Changed elite_percent to elite_ratio
        self.problem = problem
        self.population_size = population_size # Standardized attribute name
        self.elite_ratio = elite_ratio # Standardized attribute name (was elite_percent)
        self.max_iterations = max_iterations

        if hasattr(problem, 'num_genes'): # Prioritize num_genes
            self.num_genes = problem.num_genes
        elif hasattr(problem, 'num_items'):
            self.num_genes = problem.num_items # Use num_items as num_genes
        else:
            raise ValueError("Problem object must have 'num_items' or 'num_genes' attribute.")

        self.tree_root_node = None 
        self.timing_info = {'total_time': 0.0, 'iteration_times': []}
        self.prob_marginal = np.array([]) 
        self.elite_samples = np.array([]) 
        self.random_seed = None
        self.fitness_history = [] 
        self._current_fitness_scores = np.array([]) # To store current population's fitness scores

    def initialize_population(self):
        if self.population_size == 0: return np.array([]).reshape(0, self.num_genes) # Use self.num_genes
        if self.num_genes == 0: return np.array([]).reshape(self.population_size, 0)
        return np.random.randint(2, size=(self.population_size, self.num_genes)) # Use self.num_genes

    def evaluate_population(self, population):
        if population.shape[0] == 0: return np.array([])
        if not callable(getattr(self.problem, "fitness", None)):
            raise ValueError("Problem object must have a callable 'fitness' method.")
        return np.array([self.problem.fitness(individual) for individual in population])

    def select_elite(self, population, fitness_scores):
        if len(fitness_scores) == 0 or population.shape[0] == 0:
            return np.array([]).reshape(0, self.num_genes) # Use self.num_genes

        # elite_ratio is now used here
        elite_size = max(1, int(self.elite_ratio * self.population_size)) 
        elite_size = min(elite_size, len(population)) 

        if elite_size == 0 : 
            return np.array([]).reshape(0, self.num_genes) # Use self.num_genes

        elite_indices = np.argsort(fitness_scores)[-elite_size:] 
        return population[elite_indices]

    def compute_mutual_information(self, elite_data):
        num_vars = self.num_genes # Use self.num_genes
        mutual_info = np.zeros((num_vars, num_vars))

        if elite_data.shape[0] < 2 or num_vars < 2:
            return mutual_info 

        for i in range(num_vars):
            for j in range(i + 1, num_vars): 
                p_i1 = np.mean(elite_data[:, i])
                p_i0 = 1.0 - p_i1
                p_j1 = np.mean(elite_data[:, j])
                p_j0 = 1.0 - p_j1

                if abs(p_i0) < 1e-9 or abs(p_i1) < 1e-9 or abs(p_j0) < 1e-9 or abs(p_j1) < 1e-9:
                    mutual_info[i,j] = mutual_info[j,i] = 0.0
                    continue

                p_11 = np.mean((elite_data[:, i] == 1) & (elite_data[:, j] == 1))
                p_10 = np.mean((elite_data[:, i] == 1) & (elite_data[:, j] == 0))
                p_01 = np.mean((elite_data[:, i] == 0) & (elite_data[:, j] == 1))
                p_00 = np.mean((elite_data[:, i] == 0) & (elite_data[:, j] == 0))

                mi = 0.0
                if p_00 > 1e-9 and (p_i0 * p_j0) > 1e-9: mi += p_00 * np.log2(p_00 / (p_i0 * p_j0))
                if p_01 > 1e-9 and (p_i0 * p_j1) > 1e-9: mi += p_01 * np.log2(p_01 / (p_i0 * p_j1))
                if p_10 > 1e-9 and (p_i1 * p_j0) > 1e-9: mi += p_10 * np.log2(p_10 / (p_i1 * p_j0))
                if p_11 > 1e-9 and (p_i1 * p_j1) > 1e-9: mi += p_11 * np.log2(p_11 / (p_i1 * p_j1))

                mutual_info[i, j] = mutual_info[j, i] = mi
        return mutual_info

    def build_dependency_tree(self, mutual_info):
        num_vars = self.num_genes # Use self.num_genes
        tree = defaultdict(list) 

        if num_vars == 0:
            self.tree_root_node = None
            return tree
        if num_vars == 1:
            self.tree_root_node = 0 
            return tree 

        selected_nodes = [False] * num_vars 

        if self.prob_marginal.size == num_vars and num_vars > 0: 
            with np.errstate(divide='ignore', invalid='ignore'): 
                p1 = self.prob_marginal
                p0 = 1.0 - p1
                term1 = np.where(p1 > 1e-9, p1 * np.log2(p1), 0) 
                term0 = np.where(p0 > 1e-9, p0 * np.log2(p0), 0) 
                entropies = -(term1 + term0)
            entropies = np.nan_to_num(entropies) 
            self.tree_root_node = np.argmax(entropies) if entropies.size > 0 else 0
        else:
            print_debug("MIMIC_O2 Warning: Marginal probabilities not set or incorrect size for entropy calculation. Defaulting root to 0.")
            self.tree_root_node = 0 if num_vars > 0 else None 

        if num_vars > 0: 
            if not (0 <= self.tree_root_node < num_vars): 
                self.tree_root_node = 0 
            selected_nodes[self.tree_root_node] = True
        else: 
             self.tree_root_node = None 

        num_edges_in_tree = 0

        while num_edges_in_tree < num_vars - 1 and num_vars > 1 : 
            max_mi_edge = -np.inf
            best_s_node, best_ns_node = -1, -1 

            for s_idx in range(num_vars): 
                if selected_nodes[s_idx]:
                    for ns_idx in range(num_vars): 
                        if not selected_nodes[ns_idx]:
                            if mutual_info[s_idx, ns_idx] > max_mi_edge:
                                max_mi_edge = mutual_info[s_idx, ns_idx]
                                best_s_node, best_ns_node = s_idx, ns_idx

            if best_s_node != -1 and best_ns_node != -1 : 
                tree[best_s_node].append(best_ns_node) 
                selected_nodes[best_ns_node] = True
                num_edges_in_tree += 1
            else:
                if num_vars > 1: 
                    print_debug(f"MIMIC_O2 Warning: Could not complete MST. Edges found: {num_edges_in_tree}/{num_vars-1}")
                break
        return tree

    def sample_new_population(self, tree):
        new_population = np.zeros((self.population_size, self.num_genes), dtype=int) # Use self.num_genes
        if self.population_size == 0 or self.num_genes == 0: # Use self.num_genes
            return new_population 

        if self.prob_marginal.size != self.num_genes: # Use self.num_genes
            print_debug("MIMIC_O2 Warning: Marginal probabilities incorrect size for sampling. Defaulting to 0.5.")
            self.prob_marginal = np.full(self.num_genes, 0.5) # Use self.num_genes

        actual_root = self.tree_root_node
        if self.num_genes > 0: # Use self.num_genes
            if actual_root is None or not (0 <= actual_root < self.num_genes):  # Use self.num_genes
                print_debug(f"MIMIC_O2 Warning: Invalid root node {actual_root}. Defaulting to 0.")
                actual_root = 0 
        else: 
            actual_root = None

        for k in range(self.population_size): 
            sample = np.full(self.num_genes, -1, dtype=int) # Use self.num_genes

            if self.num_genes > 0 and actual_root is not None:  # Use self.num_genes
                sample[actual_root] = 1 if np.random.rand() < self.prob_marginal[actual_root] else 0

            queue = [actual_root] if self.num_genes > 0 and actual_root is not None else [] # Use self.num_genes
            
            visited_for_sampling = np.full(self.num_genes, False, dtype=bool) # Use self.num_genes
            if actual_root is not None and self.num_genes > 0: # Use self.num_genes
                 visited_for_sampling[actual_root] = True

            head = 0
            while head < len(queue):
                parent_node = queue[head]; head += 1
                children_of_parent = tree.get(parent_node, []) 

                for child_node in children_of_parent:
                    if sample[child_node] == -1: 
                        cond_prob_child_is_1 = self.prob_marginal[child_node] 

                        if self.elite_samples.shape[0] > 0: 
                            idx_parent_match = (self.elite_samples[:, parent_node] == sample[parent_node])
                            elite_subset_parent_match = self.elite_samples[idx_parent_match]

                            if elite_subset_parent_match.shape[0] > 0:
                                count_child_is_1_given_parent_val = np.sum(elite_subset_parent_match[:, child_node])
                                cond_prob_child_is_1 = count_child_is_1_given_parent_val / elite_subset_parent_match.shape[0]

                        sample[child_node] = 1 if np.random.rand() < cond_prob_child_is_1 else 0
                        visited_for_sampling[child_node] = True
                        if child_node not in queue: 
                             queue.append(child_node) 
            
            for i in range(self.num_genes): # Use self.num_genes
                if sample[i] == -1: 
                    print_debug(f"MIMIC_O2: Node {i} not sampled via tree. Using marginal.")
                    sample[i] = 1 if np.random.rand() < self.prob_marginal[i] else 0
            new_population[k] = sample

        if self.population_size > 1 and self.num_genes > 0 and new_population.shape[0] > 0 and np.unique(new_population, axis=0).shape[0] == 1: # Use self.num_genes
            print_debug("MIMIC_O2: Population converged. Injecting diversity.")
            num_to_reset = max(1, self.population_size // 10) 
            indices_to_reset = np.random.choice(self.population_size, size=num_to_reset, replace=False)
            new_population[indices_to_reset] = np.random.randint(2, size=(num_to_reset, self.num_genes)) # Use self.num_genes
        return new_population

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

        init_start_time = time.time()

        overall_best_solution = None
        overall_best_fitness = -np.inf
        
        population = self.initialize_population() 
        self._current_fitness_scores = self.evaluate_population(population) # Store initial fitness scores

        if self.num_genes == 0:  # Use self.num_genes
            overall_best_solution = np.array([]).reshape(0,)
            overall_best_fitness = self.problem.fitness(overall_best_solution) if callable(getattr(self.problem, "fitness", None)) else 0.0
        elif self._current_fitness_scores.size > 0: 
            current_iter_best_idx = np.argmax(self._current_fitness_scores)
            overall_best_fitness = self._current_fitness_scores[current_iter_best_idx]
            overall_best_solution = population[current_iter_best_idx].copy()
        
        self.fitness_history.append(overall_best_fitness)
        self.timing_info['iteration_times'].append(time.time() - init_start_time)
        print_debug(f"[MIMIC_O2] Initial Best Fitness: {overall_best_fitness:.4f}")
        if self._current_fitness_scores.size > 0:
             print_debug(f"[MIMIC_O2] Initial Avg Fitness: {np.mean(self._current_fitness_scores):.4f}")


        if self.population_size == 0: 
            self.timing_info['total_time'] = time.time() - overall_start_time
            return overall_best_solution, overall_best_fitness, self.fitness_history, self.timing_info

        for iteration in range(self.max_iterations):
            iter_start_time = time.time()

            self.elite_samples = self.select_elite(population, self._current_fitness_scores)

            # Stats from the population *before* resampling for this iteration
            prev_pop_avg_fitness = -np.inf
            prev_pop_best_fitness = -np.inf 
            if self._current_fitness_scores.size > 0: 
                prev_pop_best_fitness = np.max(self._current_fitness_scores)
                prev_pop_avg_fitness = np.mean(self._current_fitness_scores)


            if self.elite_samples.shape[0] == 0 and self.population_size > 0 :
                print_debug(f"MIMIC_O2 Iter {iteration+1}: No elite samples found. Re-initializing population.")
                population = self.initialize_population()
                self.prob_marginal = np.full(self.num_genes, 0.5) if self.num_genes > 0 else np.array([]) # Use self.num_genes
                self.tree_root_node = 0 if self.num_genes > 0 else None  # Use self.num_genes
            elif self.num_genes > 0 and self.population_size > 0:  # Use self.num_genes
                self.prob_marginal = np.mean(self.elite_samples, axis=0) if self.elite_samples.shape[0] > 0 else np.full(self.num_genes, 0.5) # Use self.num_genes
                mutual_info = self.compute_mutual_information(self.elite_samples)
                tree = self.build_dependency_tree(mutual_info) 
                population = self.sample_new_population(tree)
            
            self._current_fitness_scores = self.evaluate_population(population) # Fitness of NEWLY SAMPLED population

            # Stats for the current, newly sampled population
            curr_gen_best_fitness = -np.inf 
            curr_gen_avg_fitness = -np.inf  

            if self._current_fitness_scores.size > 0:
                current_gen_best_idx = np.argmax(self._current_fitness_scores)
                curr_gen_best_fitness = self._current_fitness_scores[current_gen_best_idx]
                curr_gen_avg_fitness = np.mean(self._current_fitness_scores)
                if curr_gen_best_fitness > overall_best_fitness:
                    overall_best_fitness = curr_gen_best_fitness
                    overall_best_solution = population[current_gen_best_idx].copy()
            elif self.num_genes == 0:  # Use self.num_genes
                curr_gen_best_fitness = overall_best_fitness 
                curr_gen_avg_fitness = overall_best_fitness


            self.fitness_history.append(overall_best_fitness) 
            iter_time_taken = time.time() - iter_start_time
            self.timing_info['iteration_times'].append(iter_time_taken)

            if (iteration + 1) % 10 == 0 or iteration == self.max_iterations -1:
                print_debug(f"[MIMIC_O2] Iteration {iteration+1}/{self.max_iterations}: "
                            # f"Prev Pop Best = {prev_pop_best_fitness:.4f}, " # Optional: Best of population before resampling
                            # f"Prev Pop Avg = {prev_pop_avg_fitness:.4f}, "    # Optional: Avg of population before resampling
                            f"Curr Pop Best = {curr_gen_best_fitness:.4f}, "      
                            f"Curr Pop Avg = {curr_gen_avg_fitness:.4f}, "      
                            f"Overall Best = {overall_best_fitness:.4f}, Time: {iter_time_taken:.2f}s")

        self.timing_info['total_time'] = time.time() - overall_start_time
        print_debug(f"[MIMIC_O2] Optimization finished. Total time: {self.timing_info['total_time']:.2f}s")
        print_debug(f"[MIMIC_O2] Final Best Fitness: {overall_best_fitness:.4f}")
        if self._current_fitness_scores is not None and self._current_fitness_scores.size > 0:
            final_avg_fitness = np.mean(self._current_fitness_scores)
            print_debug(f"[MIMIC_O2] Final Avg Fitness of last population: {final_avg_fitness:.4f}")
        return overall_best_solution, overall_best_fitness, self.fitness_history, self.timing_info
