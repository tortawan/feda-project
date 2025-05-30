# feda_project/feda_algorithm/rf_mimic.py
"""
Implements the Forest-guided Estimation of Distributions Algorithm (FEDA),
also referred to as RF-MIMIC.
"""

import numpy as np
import time
import random
from sklearn.ensemble import RandomForestClassifier
from utils.debugging import print_debug

class RF_MIMIC:
    """
    Implements the Forest-guided Estimation of Distributions Algorithm (FEDA).
    Standardized attributes: self.population_size, self.elite_ratio, self.num_genes
    """
    def __init__(self, problem, population_size: int = 100,
                 max_iterations: int = 100, elite_ratio: float = 0.2,
                 rf_params: dict = None, branch_alpha: float = 0.1):
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
        self.elite_ratio = elite_ratio 
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
        self.avg_fitness_history = [] 
        self.timing_info = {'total_time': 0.0, 'iteration_times':[]}
        self.rf_model_for_inspection = None

    def _initialize_population(self) -> np.ndarray:
        if self.num_genes == 0:
            return np.array([]).reshape(self.population_size, 0) 
        return np.random.randint(0, 2, size=(self.population_size, self.num_genes), dtype=int) 

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        if population.shape[0] == 0:
            return np.array([])
        return np.array([self.fitness_fn(ind) for ind in population])

    def _select_elite(self, population: np.ndarray, fitnesses: np.ndarray) -> tuple:
        pop_size_current = population.shape[0]
        if pop_size_current == 0:
            return (np.array([]).reshape(0, self.num_genes), np.array([]),
                    np.array([]).reshape(0, self.num_genes), np.array([]))
        if self.elite_ratio < 1.0: 
            elite_count = int(np.floor(pop_size_current * self.elite_ratio))
        else: 
            elite_count = int(self.elite_ratio)
        elite_count = max(1 if pop_size_current > 0 else 0, elite_count)
        elite_count = min(elite_count, pop_size_current)
        if elite_count == 0 and pop_size_current > 0:
            elite_count = 1
        if fitnesses.size == 0 and pop_size_current > 0 :
            print_debug("[RF_MIMIC] Warn: Fitnesses empty in _select_elite despite population. Re-evaluating.")
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
        current_rf_params = self.rf_params.copy()
        current_rf_params['random_state'] = self.random_seed
        if elite_population.shape[0] > 0:
            max_min_leaf = max(1, elite_population.shape[0] // 2 if elite_population.shape[0] > 1 else 1)
            if current_rf_params.get('min_samples_leaf', 1) > max_min_leaf:
                current_rf_params['min_samples_leaf'] = max_min_leaf
        rf_model_unfitted = RandomForestClassifier(**current_rf_params)
        if elite_population.shape[0] == 0 or nonelite_population.shape[0] == 0:
            print_debug(f"[RF_MIMIC] Warning: RF training with insufficient distinct classes. Elites: {elite_population.shape[0]}, Non-elites: {nonelite_population.shape[0]}")
            return rf_model_unfitted
        X_train = np.vstack((elite_population, nonelite_population))
        y_train = np.concatenate((
            np.ones(elite_population.shape[0], dtype=int),
            np.zeros(nonelite_population.shape[0], dtype=int)
        ))
        if len(np.unique(y_train)) < 2:
            print_debug(f"[RF_MIMIC] Warning: Training RF with only one class. Unique labels: {np.unique(y_train)}")
            return rf_model_unfitted
        if X_train.shape[0] == 0:
            print_debug("[RF_MIMIC] Warning: No data to train Random Forest (X_train is empty).")
            return rf_model_unfitted
        try:
            rf_model_unfitted.fit(X_train, y_train)
            self.rf_model_for_inspection = rf_model_unfitted
        except ValueError as e:
            print_debug(f"[RF_MIMIC] Warning: RandomForestClassifier fitting error: {e}. Returning unfitted model.")
            return RandomForestClassifier(**current_rf_params)
        return rf_model_unfitted

    def _sample_new_population(self, rf_model: RandomForestClassifier,
                               elite_population: np.ndarray) -> np.ndarray:
        new_population_list =[]
        if self.population_size == 0: return np.array([]).reshape(0, self.num_genes) 
        if self.num_genes == 0: return np.array([]).reshape(self.population_size, 0) 
        if not hasattr(rf_model, 'estimators_') or not rf_model.estimators_:
            print_debug("[RF_MIMIC] Warning: RF model not trained or has no trees. Falling back.")
            if elite_population.shape[0] > 0 and self.population_size > 0: 
                return self._sample_new_population_from_elites_only(elite_population, target_size=self.population_size) 
            return self._initialize_population()
        trees = rf_model.estimators_
        n_trees = len(trees)
        if n_trees == 0:
            print_debug("[RF_MIMIC] Warning: RF model has no trees (estimators_ is empty). Falling back.")
            if elite_population.shape[0] > 0 and self.population_size > 0: 
                return self._sample_new_population_from_elites_only(elite_population, target_size=self.population_size) 
            return self._initialize_population()
        samples_per_tree = [self.population_size // n_trees] * n_trees 
        for i in range(self.population_size % n_trees): samples_per_tree[i] += 1 
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
                    print_debug(f"[RF_MIMIC] Warning: Tree {t_index} has no tree_struct (None). Skipping.")
                    continue
            except AttributeError:
                print_debug(f"[RF_MIMIC] Warning: Tree {t_index} does not have 'tree_' attribute. Skipping.")
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
                        print_debug("[RF_MIMIC] Warning: Traversed to a non-existent child node (-1). Breaking path.")
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
        if not new_population_list and self.population_size > 0: 
            print_debug("[RF_MIMIC] Warning: New population list empty after RF sampling. Falling back.")
            if elite_population.shape[0] > 0:
                return self._sample_new_population_from_elites_only(elite_population, target_size=self.population_size) 
            return self._initialize_population()
        final_population_array = np.array(new_population_list, dtype=int) if new_population_list else np.array([]).reshape(0, self.num_genes)
        if final_population_array.shape[0] < self.population_size and self.population_size > 0: 
            num_needed = self.population_size - final_population_array.shape[0] 
            print_debug(f"[RF_MIMIC] Population short by {num_needed} after RF sampling. Filling...")
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
        if final_population_array.shape[0] > self.population_size: 
            final_population_array = final_population_array[:self.population_size] 
        if self.num_genes == 0 and self.population_size > 0: 
            return np.array([]).reshape(self.population_size, 0) 
        if self.population_size == 0: 
            return np.array([]).reshape(0, self.num_genes)
        return final_population_array

    def _sample_new_population_from_elites_only(self, elite_population: np.ndarray, target_size: int) -> np.ndarray:
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
        self.avg_fitness_history = [] 
        self.timing_info = {'total_time': 0.0, 'iteration_times':[]}
        self.best_solution = None 
        self.best_fitness = -np.inf 
        if self.population_size == 0: 
            print_debug("[RF_MIMIC] Population size is 0. Returning.")
            self.timing_info['total_time'] = time.time() - overall_start_time
            if self.num_genes == 0:
                empty_sol_fitness = self.fitness_fn(np.array([]).reshape(0,)) 
                return np.array([]).reshape(0,), empty_sol_fitness, [], [empty_sol_fitness], self.timing_info 
            else:
                return None, -np.inf, [], [], self.timing_info
        init_time_start = time.time()
        self.population = self._initialize_population()
        current_avg_fitness_for_iteration = -np.inf 
        if self.population.shape[0] > 0: 
            self.fitnesses = self._evaluate_population(self.population)
            if self.fitnesses.size > 0: 
                 current_avg_fitness_for_iteration = np.mean(self.fitnesses)
        else: 
            self.fitnesses = np.array([])
            if self.num_genes == 0: 
                current_avg_fitness_for_iteration = self.fitness_fn(np.array([]).reshape(0,))
        if self.fitnesses is not None and self.fitnesses.size > 0:
            current_best_idx = np.argmax(self.fitnesses)
            self.best_solution = self.population[current_best_idx].copy()
            self.best_fitness = self.fitnesses[current_best_idx]
        else: 
            if self.num_genes == 0 and self.population_size > 0:  
                self.best_solution = np.array([]).reshape(0,)
                self.best_fitness = self.fitness_fn(self.best_solution)
                current_avg_fitness_for_iteration = self.best_fitness 
        self.fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(current_avg_fitness_for_iteration) 
        self.timing_info['iteration_times'].append(time.time() - init_time_start)
        print_debug(f"[RF_MIMIC] Initial Best Fitness: {self.best_fitness:.4f}")
        if self.fitnesses is not None and self.fitnesses.size > 0:
            initial_avg_fitness = np.mean(self.fitnesses) 
            print_debug(f"[RF_MIMIC] Initial Avg Fitness: {initial_avg_fitness:.4f}")
        elif self.num_genes == 0 and self.population_size > 0: 
            print_debug(f"[RF_MIMIC] Initial Avg Fitness: {current_avg_fitness_for_iteration:.4f}")
        for iteration in range(self.max_iterations):
            iter_start_time = time.time()
            elite_pop, _, nonelite_pop, _ = \
                self._select_elite(self.population, self.fitnesses)
            action_taken = "initial_selection_done"
            if elite_pop.shape[0] == 0 and self.population_size > 0: 
                print_debug(f"[RF_MIMIC] Iter {iteration + 1}: No elite samples. Re-initializing.")
                self.population = self._initialize_population()
                action_taken = "reinit_due_to_no_elites"
            elif nonelite_pop.shape[0] == 0 and elite_pop.shape[0] == self.population_size and self.population_size > 0: 
                print_debug(f"[RF_MIMIC] Iter {iteration + 1}: All samples are elite. Sampling from elites.")
                self.population = self._sample_new_population_from_elites_only(elite_pop, target_size=self.population_size) 
                action_taken = "sample_from_all_elites"
            elif self.population_size > 0:  
                rf_model = self._train_random_forest(elite_pop, nonelite_pop)
                if hasattr(rf_model, 'estimators_') and rf_model.estimators_ and len(rf_model.estimators_) > 0:
                    self.population = self._sample_new_population(rf_model, elite_pop)
                    action_taken = "sampled_from_trained_rf"
                else:
                    print_debug(f"[RF_MIMIC] Iter {iteration + 1}: RF model not effectively trained. Fallback.")
                    if elite_pop.shape[0] > 0:
                        self.population = self._sample_new_population_from_elites_only(elite_pop, target_size=self.population_size) 
                        action_taken = "rf_train_fail_fallback_elites"
                    else:
                        self.population = self._initialize_population()
                        action_taken = "rf_train_fail_fallback_random"
            if self.population_size > 0 and (self.population.shape[0]!= self.population_size or \
                                      (self.num_genes == 0 and self.population.shape[1]!=0)): 
                print_debug(f"[RF_MIMIC] Iter {iteration + 1}: Pop size {self.population.shape} not {self.population_size}x{self.num_genes} after {action_taken}. Adjusting.") 
                if self.population.shape[0] < self.population_size: 
                    num_needed = self.population_size - self.population.shape[0] 
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
                elif self.population.shape[0] > self.population_size: 
                    self.population = self.population[:self.population_size] 
                if self.num_genes == 0 and self.population.shape!= (self.population_size, 0):  
                    self.population = np.array([]).reshape(self.population_size, 0) 
            if self.population_size > 0 and self.population.shape[0] == 0 and self.num_genes > 0 : 
                print_debug(f"[RF_MIMIC] Critical Error: Population empty at iter {iteration + 1}. Reinitializing.")
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
            elif self.num_genes == 0 and self.population_size > 0:  
                current_iter_best_fitness = self.best_fitness 
                current_iter_avg_fitness = self.best_fitness 
            self.fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(current_iter_avg_fitness) 
            iter_time_taken = time.time() - iter_start_time
            self.timing_info['iteration_times'].append(iter_time_taken)
            if (iteration + 1) % 10 == 0 or iteration == self.max_iterations - 1:
                print_debug(f"[RF_MIMIC] Iteration {iteration + 1}/{self.max_iterations}: " 
                            f"Iter Pop Best = {current_iter_best_fitness:.4f}, "
                            f"Iter Pop Avg = {current_iter_avg_fitness:.4f}, " 
                            f"Overall Best = {self.best_fitness:.4f}, Time: {iter_time_taken:.2f}s")
        self.timing_info['total_time'] = time.time() - overall_start_time
        print_debug(f"[RF_MIMIC] Optimization finished. Total time: {self.timing_info['total_time']:.2f}s") 
        print_debug(f"[RF_MIMIC] Final Best Fitness: {self.best_fitness:.4f}") 
        if self.fitnesses is not None and self.fitnesses.size > 0:
            final_avg_fitness = np.mean(self.fitnesses)
            print_debug(f"[RF_MIMIC] Final Avg Fitness of last population: {final_avg_fitness:.4f}") 
        return self.best_solution, self.best_fitness, self.fitness_history, self.avg_fitness_history, self.timing_info
