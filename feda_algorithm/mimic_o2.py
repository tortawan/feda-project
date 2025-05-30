# feda_project/feda_algorithm/mimic_o2.py
"""
Implements the MIMIC algorithm using a dependency tree from mutual information.
"""

import numpy as np
import time
import random
from collections import defaultdict
from utils.debugging import print_debug

class MIMIC_O2:
    """
    Implements the MIMIC algorithm using a dependency tree.
    Standardized attributes: self.population_size, self.elite_ratio, self.num_genes
    """
    def __init__(self, problem, population_size=500, elite_ratio=0.2, max_iterations=100): 
        self.problem = problem
        self.population_size = population_size 
        self.elite_ratio = elite_ratio 
        self.max_iterations = max_iterations
        if hasattr(problem, 'num_genes'): 
            self.num_genes = problem.num_genes
        elif hasattr(problem, 'num_items'):
            self.num_genes = problem.num_items 
        else:
            raise ValueError("Problem object must have 'num_items' or 'num_genes' attribute.")
        self.tree_root_node = None 
        self.timing_info = {'total_time': 0.0, 'iteration_times': []}
        self.prob_marginal = np.array([]) 
        self.elite_samples = np.array([]) 
        self.random_seed = None
        self.fitness_history = [] 
        self.avg_fitness_history = [] 
        self._current_fitness_scores = np.array([]) 

    def initialize_population(self):
        if self.population_size == 0: return np.array([]).reshape(0, self.num_genes) 
        if self.num_genes == 0: return np.array([]).reshape(self.population_size, 0)
        return np.random.randint(2, size=(self.population_size, self.num_genes)) 

    def evaluate_population(self, population):
        if population.shape[0] == 0: return np.array([])
        if not callable(getattr(self.problem, "fitness", None)):
            raise ValueError("Problem object must have a callable 'fitness' method.")
        return np.array([self.problem.fitness(individual) for individual in population])

    def select_elite(self, population, fitness_scores):
        if len(fitness_scores) == 0 or population.shape[0] == 0:
            return np.array([]).reshape(0, self.num_genes) 
        elite_size = max(1, int(self.elite_ratio * self.population_size)) 
        elite_size = min(elite_size, len(population)) 
        if elite_size == 0 : 
            return np.array([]).reshape(0, self.num_genes) 
        elite_indices = np.argsort(fitness_scores)[-elite_size:] 
        return population[elite_indices]

    def compute_mutual_information(self, elite_data):
        num_vars = self.num_genes 
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
        num_vars = self.num_genes 
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
            print_debug("[MIMIC_O2] Warning: Marginal probabilities not set or incorrect size for entropy calculation. Defaulting root to 0.")
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
                    print_debug(f"[MIMIC_O2] Warning: Could not complete MST. Edges found: {num_edges_in_tree}/{num_vars-1}")
                break
        return tree

    def sample_new_population(self, tree):
        new_population = np.zeros((self.population_size, self.num_genes), dtype=int) 
        if self.population_size == 0 or self.num_genes == 0: 
            return new_population 
        if self.prob_marginal.size != self.num_genes: 
            print_debug("[MIMIC_O2] Warning: Marginal probabilities incorrect size for sampling. Defaulting to 0.5.")
            self.prob_marginal = np.full(self.num_genes, 0.5) 
        actual_root = self.tree_root_node
        if self.num_genes > 0: 
            if actual_root is None or not (0 <= actual_root < self.num_genes):  
                print_debug(f"[MIMIC_O2] Warning: Invalid root node {actual_root}. Defaulting to 0.")
                actual_root = 0 
        else: 
            actual_root = None
        for k in range(self.population_size): 
            sample = np.full(self.num_genes, -1, dtype=int) 
            if self.num_genes > 0 and actual_root is not None:  
                sample[actual_root] = 1 if np.random.rand() < self.prob_marginal[actual_root] else 0
            queue = [actual_root] if self.num_genes > 0 and actual_root is not None else [] 
            visited_for_sampling = np.full(self.num_genes, False, dtype=bool) 
            if actual_root is not None and self.num_genes > 0: 
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
            for i in range(self.num_genes): 
                if sample[i] == -1: 
                    print_debug(f"[MIMIC_O2] Node {i} not sampled via tree. Using marginal.")
                    sample[i] = 1 if np.random.rand() < self.prob_marginal[i] else 0
            new_population[k] = sample
        if self.population_size > 1 and self.num_genes > 0 and new_population.shape[0] > 0 and np.unique(new_population, axis=0).shape[0] == 1: 
            print_debug("[MIMIC_O2] Population converged. Injecting diversity.")
            num_to_reset = max(1, self.population_size // 10) 
            indices_to_reset = np.random.choice(self.population_size, size=num_to_reset, replace=False)
            new_population[indices_to_reset] = np.random.randint(2, size=(num_to_reset, self.num_genes)) 
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
        self.avg_fitness_history = [] 
        init_start_time = time.time()
        overall_best_solution = None
        overall_best_fitness = -np.inf
        population = self.initialize_population() 
        self._current_fitness_scores = self.evaluate_population(population) 
        initial_avg_fitness_val = -np.inf
        if self.num_genes == 0:  
            overall_best_solution = np.array([]).reshape(0,)
            overall_best_fitness = self.problem.fitness(overall_best_solution) if callable(getattr(self.problem, "fitness", None)) else 0.0
            initial_avg_fitness_val = overall_best_fitness 
        elif self._current_fitness_scores.size > 0: 
            current_iter_best_idx = np.argmax(self._current_fitness_scores)
            overall_best_fitness = self._current_fitness_scores[current_iter_best_idx]
            overall_best_solution = population[current_iter_best_idx].copy()
            initial_avg_fitness_val = np.mean(self._current_fitness_scores)
        self.fitness_history.append(overall_best_fitness)
        self.avg_fitness_history.append(initial_avg_fitness_val) 
        self.timing_info['iteration_times'].append(time.time() - init_start_time)
        print_debug(f"[MIMIC_O2] Initial Best Fitness: {overall_best_fitness:.4f}")
        if self._current_fitness_scores.size > 0: 
             print_debug(f"[MIMIC_O2] Initial Avg Fitness: {initial_avg_fitness_val:.4f}")
        if self.population_size == 0: 
            self.timing_info['total_time'] = time.time() - overall_start_time
            return overall_best_solution, overall_best_fitness, self.fitness_history, self.avg_fitness_history, self.timing_info
        for iteration in range(self.max_iterations):
            iter_start_time = time.time()
            self.elite_samples = self.select_elite(population, self._current_fitness_scores)
            if self.elite_samples.shape[0] == 0 and self.population_size > 0 :
                print_debug(f"[MIMIC_O2] Iter {iteration+1}: No elite samples found. Re-initializing population.")
                population = self.initialize_population()
                self.prob_marginal = np.full(self.num_genes, 0.5) if self.num_genes > 0 else np.array([]) 
                self.tree_root_node = 0 if self.num_genes > 0 else None  
            elif self.num_genes > 0 and self.population_size > 0:  
                self.prob_marginal = np.mean(self.elite_samples, axis=0) if self.elite_samples.shape[0] > 0 else np.full(self.num_genes, 0.5) 
                mutual_info = self.compute_mutual_information(self.elite_samples)
                tree = self.build_dependency_tree(mutual_info) 
                population = self.sample_new_population(tree)
            self._current_fitness_scores = self.evaluate_population(population) 
            curr_gen_best_fitness = -np.inf 
            curr_gen_avg_fitness = -np.inf  
            if self._current_fitness_scores.size > 0:
                current_gen_best_idx = np.argmax(self._current_fitness_scores)
                curr_gen_best_fitness = self._current_fitness_scores[current_gen_best_idx]
                curr_gen_avg_fitness = np.mean(self._current_fitness_scores)
                if curr_gen_best_fitness > overall_best_fitness:
                    overall_best_fitness = curr_gen_best_fitness
                    overall_best_solution = population[current_gen_best_idx].copy()
            elif self.num_genes == 0:  
                curr_gen_best_fitness = overall_best_fitness 
                curr_gen_avg_fitness = overall_best_fitness
            self.fitness_history.append(overall_best_fitness) 
            self.avg_fitness_history.append(curr_gen_avg_fitness) 
            iter_time_taken = time.time() - iter_start_time
            self.timing_info['iteration_times'].append(iter_time_taken)
            if (iteration + 1) % 10 == 0 or iteration == self.max_iterations -1:
                print_debug(f"[MIMIC_O2] Iteration {iteration+1}/{self.max_iterations}: "
                            f"Curr Pop Best = {curr_gen_best_fitness:.4f}, "      
                            f"Curr Pop Avg = {curr_gen_avg_fitness:.4f}, "      
                            f"Overall Best = {overall_best_fitness:.4f}, Time: {iter_time_taken:.2f}s")
        self.timing_info['total_time'] = time.time() - overall_start_time
        print_debug(f"[MIMIC_O2] Optimization finished. Total time: {self.timing_info['total_time']:.2f}s")
        print_debug(f"[MIMIC_O2] Final Best Fitness: {overall_best_fitness:.4f}")
        if self._current_fitness_scores is not None and self._current_fitness_scores.size > 0:
            final_avg_fitness = np.mean(self._current_fitness_scores)
            print_debug(f"[MIMIC_O2] Final Avg Fitness of last population: {final_avg_fitness:.4f}")
        return overall_best_solution, overall_best_fitness, self.fitness_history, self.avg_fitness_history, self.timing_info
