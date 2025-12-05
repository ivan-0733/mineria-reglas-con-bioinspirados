""" 
This module implements the MOEA/D algorithm for multi-objective optimization
applied to Association Rule Mining (ARM).

It integrates custom components:
- AdaptiveMOEAD: Extends pymoo's MOEA/D with 1/5 Rule for adaptive mutation.
- ARMProblem: Defines the optimization problem (objectives, evaluation).
- Custom Operators: Mutation, Crossover, Sampling.
"""

import numpy as np
import math
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.tchebicheff import Tchebicheff as Tcheb
from pymoo.decomposition.weighted_sum import WeightedSum as WS
from pymoo.termination.default import DefaultMultiObjectiveTermination

from src.individual import Individual
from src.crossover import DiploidNPointCrossover
from src.mutation import ARMMutation
from src.validator import ARMValidator
from src.metrics import ARMMetrics
from src.initialization import ARMSampling
from src.loggers import DiscardedRulesLogger

def get_H_from_N(N, M):
    """
    Calculates the number of partitions H for Das-Dennis weights 
    given a target population size N and number of objectives M.
    Returns H such that nCr(H+M-1, M-1) is closest to N.
    """
    H = 1
    while True:
        count = math.comb(H + M - 1, M - 1)
        if count >= N:
            # Check if previous H was closer
            prev_count = math.comb((H - 1) + M - 1, M - 1)
            if abs(prev_count - N) < abs(count - N):
                return H - 1
            return H
        H += 1

from pymoo.core.population import Population

import sys

class AdaptiveMOEAD(MOEAD):
    """
    Custom MOEA/D implementation that adapts mutation and crossover probabilities
    based on the 1/5 success rule (Rechenberg).
    """
    def __init__(self, mutation_adapter_config, crossover_adapter_config, n_replace=2, prob_neighbor_mating=0.9, **kwargs):
        super().__init__(prob_neighbor_mating=prob_neighbor_mating, **kwargs)
        self.mut_config = mutation_adapter_config
        self.cx_config = crossover_adapter_config
        self.n_replace = n_replace
        self.prob_neighbor_mating = prob_neighbor_mating
        self.current_gen = 0
        
        # Mutation Config
        self.mut_min = self.mut_config['min']
        self.mut_max = self.mut_config['max']
        
        # Crossover Config
        self.cx_min = self.cx_config['min']
        self.cx_max = self.cx_config['max']
        
        # Stats
        self.success_history = []

    def _infill(self):
        return None

    def _advance(self, infills=None):
        self._next()

    def _next(self):
        # 1. Snapshot current population (X) to detect changes
        old_X = self.pop.get("X").copy()
        
        # 2. Manual MOEA/D Step (Bypassing pymoo generator)
        pop = self.pop
        
        # Initialize ideal point if needed
        if not hasattr(self, 'ideal_point') or self.ideal_point is None:
            self.ideal_point = np.min(pop.get("F"), axis=0)

        # Random permutation
        perm = np.random.permutation(len(pop))
        
        for i in perm:
            # a) Select Neighbors
            nbs = self.neighbors[i]
            
            # b) Mating Selection
            if np.random.random() < self.prob_neighbor_mating:
                parent_indices = np.random.choice(nbs, 2, replace=False)
            else:
                parent_indices = np.random.choice(len(pop), 2, replace=False)
            
            parents = pop[parent_indices]
            
            # c) Mating (Crossover + Mutation)
            # Crossover
            p_X = np.array([parents[0].X, parents[1].X]) # (2, n_var)
            p_X_input = p_X[None, :, :] # (1, 2, n_var)
            
            # Call _do directly to bypass pymoo's object handling
            # Crossover returns (n_matings, n_parents, n_var) -> (1, 2, n_var)
            off_X_pair = self.mating.crossover._do(self.problem, p_X_input)[0] # (2, n_var)
            
            # Process BOTH offspring (User Requirement)
            for off_X_raw in off_X_pair:
                # Mutation
                # Call _do directly
                off_X_mut = self.mating.mutation._do(self.problem, off_X_raw[None, :])[0] # (n_var,)
                
                # d) Evaluation
                # Create temp population for evaluation
                off_pop = Population.new(X=np.array([off_X_mut]))
                self.evaluator.eval(self.problem, off_pop)
                off = off_pop[0]

                # Check for duplicates in current population
                # This prevents the population from collapsing to a single individual
                current_X = pop.get("X")
                if np.any(np.all(current_X == off.X, axis=1)):
                    continue
                
                # e) Update Ideal Point
                self.ideal_point = np.min(np.vstack([self.ideal_point, off.F]), axis=0)
                
                # f) Update Neighbors (Decomposition)
                weights = self.ref_dirs[nbs]
                
                off_fv = self.decomposition.do(off.F, weights, ideal_point=self.ideal_point)
                nbs_F = pop[nbs].get("F")
                nbs_fv = self.decomposition.do(nbs_F, weights, ideal_point=self.ideal_point)
                
                improved_idx = np.where(off_fv < nbs_fv)[0]
                
                if len(improved_idx) > self.n_replace:
                    np.random.shuffle(improved_idx)
                    improved_idx = improved_idx[:self.n_replace]
                
                for idx in improved_idx:
                    pop_idx = nbs[idx]
                    pop[pop_idx] = off
        
        self.current_gen += 1
        
        # Progress Bar
        total_gen = self.termination.n_max_gen if hasattr(self.termination, 'n_max_gen') else 300 # Fallback
        percent = (self.current_gen / total_gen) * 100
        bar_length = 30
        filled_length = int(bar_length * self.current_gen // total_gen)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\rProgress: |{bar}| {percent:.1f}% (Gen {self.current_gen}/{total_gen})')
        sys.stdout.flush()
        if self.current_gen >= total_gen:
            print() # Newline at end

        # 3. Calculate Success Rate (Replacements)
        new_X = self.pop.get("X")
        
        # Count how many individuals changed
        diffs = (old_X != new_X)
        changes_per_ind = np.sum(diffs, axis=1)
        n_replacements = np.sum(changes_per_ind > 0)
        
        success_rate = n_replacements / len(self.pop)
        self.success_history.append(success_rate)
        
        # 4. Adapt Probabilities (1/5 Rule)
        # Target success rate is around 0.2 (1/5)
        
        # Factor c = 0.9 (decrease) and 1.11 (increase)
        c = 0.9 
        
        # --- Adapt Mutation ---
        current_mut_prob = self.mating.mutation.prob
        if success_rate > 0.2:
            new_mut_prob = current_mut_prob / c # Increase exploration
        elif success_rate < 0.2:
            new_mut_prob = current_mut_prob * c # Decrease exploration
        else:
            new_mut_prob = current_mut_prob
        self.mating.mutation.prob = max(self.mut_min, min(self.mut_max, new_mut_prob))
        
        # --- Adapt Crossover ---
        # Logic: If success is high, we can afford more disruption (higher crossover).
        # If success is low, we might want to be more conservative or disruptive depending on strategy.
        # Standard 1/5 rule usually applies to step size (mutation).
        # For crossover, often high success -> keep high crossover to mix good blocks.
        # Low success -> maybe reduce crossover to avoid breaking good blocks?
        # Let's apply same logic: High success (>0.2) -> Increase Crossover.
        
        current_cx_prob = self.mating.crossover.prob
        if success_rate > 0.2:
            new_cx_prob = current_cx_prob / c
        elif success_rate < 0.2:
            new_cx_prob = current_cx_prob * c
        else:
            new_cx_prob = current_cx_prob
        self.mating.crossover.prob = max(self.cx_min, min(self.cx_max, new_cx_prob))


class ARMProblem(Problem):
    def __init__(self, metadata, supports, df, config, validator, metrics, logger):
        self.metadata = metadata
        self.supports = supports
        self.config = config
        self.objectives = config['objectives']['selected']
        self.n_obj = len(self.objectives)
        self.logger = logger
        
        # Determine number of genes from metadata
        dummy = Individual(metadata=metadata)
        self.n_var = 2 * dummy.num_genes
        
        self.validator = validator
        self.metrics = metrics

        super().__init__(n_var=self.n_var, 
                         n_obj=self.n_obj, 
                         n_ieq_constr=0, 
                         xl=0, 
                         xu=1) # Bounds not strictly used for custom sampling/mutation

    def _evaluate(self, x, out, *args, **kwargs):
        # x shape: (n_pop, n_var)
        n_pop = x.shape[0]
        F = np.zeros((n_pop, self.n_obj))
        
        for i in range(n_pop):
            ind_genome = x[i]
            
            # Extract rule items using temp individual
            temp_ind = Individual(self.metadata)
            temp_ind.X = ind_genome
            ant, con = temp_ind.get_rule_items()
            
            # 1. Validate Structure & Constraints
            is_valid, reason, _ = self.validator.validate(ant, con)
            
            if not is_valid:
                # Log invalid individual
                self.logger.log(temp_ind, f"invalid_structure:{reason}")
                
                # Penalty for invalid individuals
                # Assign a value worse than any possible valid metric
                # Metrics are [-1, 1] or [0, 1]. We minimize -Metric.
                # So valid range is [-1, 1].
                # We assign 2.0 to ensure it's dominated by everything.
                F[i, :] = 2.0
                continue

            # 2. Calculate Metrics
            # get_metrics returns (values_list, errors_dict)
            vals, errors = self.metrics.get_metrics(ant, con, self.objectives)
            
            # Extract objectives
            obj_values = []
            for val in vals:
                if val is None:
                    # Penalty for undefined metric
                    obj_values.append(2.0) 
                else:
                    # MOEA/D minimizes. If we want to maximize (e.g. Confidence), we negate.
                    # Assuming all ARM metrics are "higher is better".
                    obj_values.append(-val)
                
            F[i, :] = obj_values
        
        out["F"] = F
class MOEAD_ARM:
    """
    Wrapper class for MOEA/D algorithm applied to Association Rule Mining.
    Orchestrates the components.
    """
    def __init__(self, config, data_context):
        self.config = config
        self.data = data_context # Dict with df, supports, metadata
        
        # Initialize Components
        self.metrics = ARMMetrics(
            dataframe=self.data['df'],
            supports_dict=self.data['supports'],
            metadata=self.data['metadata']
        )
        
        self.validator = ARMValidator(
            config=self.config,
            metrics_engine=self.metrics,
            metadata=self.data['metadata']
        )
        
        self.logger = DiscardedRulesLogger()
        
        self.problem = ARMProblem(
            metadata=self.data['metadata'],
            supports=self.data['supports'],
            df=self.data['df'],
            config=self.config,
            validator=self.validator,
            metrics=self.metrics,
            logger=self.logger
        )
        
    def run(self, callback=None):
        # Algorithm Parameters
        alg_config = self.config['algorithm']
        target_pop_size = alg_config['population_size']
        n_gen = alg_config['generations']
        
        # Reference Directions (Decomposition)
        # Calculate H for Das-Dennis
        H = get_H_from_N(target_pop_size, self.problem.n_obj)
        ref_dirs = get_reference_directions("das-dennis", self.problem.n_obj, n_partitions=H)
        actual_pop_size = ref_dirs.shape[0]
        
        print(f"Initialized MOEA/D with {actual_pop_size} reference directions (Target: {target_pop_size})")
        
        # Operators
        # Get max attempts from config or default to 10000
        max_init_attempts = alg_config.get('initialization', {}).get('max_attempts', 10000)
        
        sampling = ARMSampling(
            metadata=self.data['metadata'],
            validator=self.validator,
            logger=self.logger,
            max_attempts=max_init_attempts
        )
        
        crossover = DiploidNPointCrossover(
            prob=alg_config['operators']['crossover']['probability']['initial'],
            n_points=2
        )
        
        mutation = ARMMutation(
            metadata=self.data['metadata'],
            validator=self.validator,
            logger=self.logger,
            prob=alg_config['operators']['mutation']['probability']['initial'],
            attempts=alg_config['operators']['mutation']['repair_attempts'],
            duplicate_attempts=alg_config['operators']['mutation'].get('duplicate_attempts', 0),
            active_ops=alg_config['operators']['mutation'].get('active_ops', ['extension', 'contraction', 'replacement'])
        )
        
        # Decomposition Method
        decomp_config = alg_config['decomposition']
        method = decomp_config['method'].lower()
        params = decomp_config.get('params', {})
        
        if method == 'pbi':
            decomposition = PBI(theta=params.get('theta', 5.0))
        elif method == 'tchebycheff' or method == 'tcheb':
            decomposition = Tcheb()
        elif method == 'weighted_sum' or method == 'ws':
            decomposition = WS()
        else:
            print(f"Warning: Unknown decomposition method '{method}'. Defaulting to PBI.")
            decomposition = PBI()

        # Algorithm Instance
        algorithm = AdaptiveMOEAD(
            mutation_adapter_config=alg_config['operators']['mutation']['probability'],
            crossover_adapter_config=alg_config['operators']['crossover']['probability'],
            ref_dirs=ref_dirs,
            n_neighbors=alg_config['neighborhood']['size'],
            n_replace=alg_config['neighborhood'].get('replacement_size', 2),
            decomposition=decomposition,
            prob_neighbor_mating=alg_config['neighborhood']['selection_probability'],
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )
        
        # Termination Criterion
        # Check for early stopping config
        termination_config = alg_config.get('termination', {})
        
        if termination_config.get('enabled', False):
            print(f"Early stopping enabled (ftol={termination_config.get('ftol')}, period={termination_config.get('period')})")
            termination = DefaultMultiObjectiveTermination(
                xtol=1e-8,
                cvtol=1e-6,
                ftol=termination_config.get('ftol', 0.0001),
                period=termination_config.get('period', 30),
                n_max_gen=n_gen,
                n_max_evals=1000000
            )
        else:
            termination = ('n_gen', n_gen)

        # Execution
        res = minimize(
            self.problem,
            algorithm,
            termination,
            seed=self.config['experiment']['random_seed'],
            callback=callback,
            verbose=False
        )
        
        return res

