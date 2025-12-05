import os
import json
import logging
import numpy as np
import pandas as pd
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from src.individual import Individual

class ARMCallback(Callback):
    def __init__(self, config, logger, output_dir):
        super().__init__()
        self.config = config
        self.rule_logger = logger
        self.output_dir = output_dir
        self.interval = config['algorithm'].get('logging_interval', 10)
        self.objectives = config['objectives']['selected']
        
        # Setup directories
        self.dirs = {
            'stats': os.path.join(output_dir, 'stats'),
            'populations': os.path.join(output_dir, 'populations'),
            'pareto': os.path.join(output_dir, 'pareto'),
            'discarded': os.path.join(output_dir, 'discarded')
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
            
        # Initialize stats file
        self.stats_file = os.path.join(self.dirs['stats'], 'evolution_stats.csv')
        self.stats_history = []
        
        # Hypervolume Indicator
        # We assume objectives are normalized [0,1] and we minimize negative values [-1, 0]
        # Reference point for minimization of [-1, 0] range is usually 0 (nadir) or slightly positive.
        # Since we want to measure volume from the front up to 0, ref_point=[0,0,0] works.
        self.hv_indicator = HV(ref_point=np.zeros(len(self.objectives)))
        
        # Track cumulative discarded count
        self.cumulative_discarded = 0
        self.log = logging.getLogger(__name__)

    def notify(self, algorithm):
        # Only run every 'interval' generations or at the last one
        if algorithm.n_gen % self.interval != 0:
            return

        self.log.info("Callback at gen %s | pop=%s", algorithm.n_gen, len(algorithm.pop))
            
        gen = algorithm.n_gen
        pop = algorithm.pop
        
        # 1. Calculate Statistics (on REAL values)
        # pop.F contains negated values (minimization). We negate back to get real [0,1] values.
        real_F = -pop.get("F")
        
        stats = {'generation': gen}
        
        # Objective Stats
        for i, obj_name in enumerate(self.objectives):
            values = real_F[:, i]
            stats[f'{obj_name}_min'] = np.min(values)
            stats[f'{obj_name}_max'] = np.max(values)
            stats[f'{obj_name}_mean'] = np.mean(values)
            
        # Hypervolume (Population)
        hv_value = 0.0
        try:
            hv_value = self.hv_indicator(pop.get("F"))
            stats['hypervolume'] = hv_value
        except Exception as e:
            stats['hypervolume'] = 0.0
            print(f"Warning: HV calculation failed at gen {gen}: {e}")
            
        # Track Operator Probabilities
        if hasattr(algorithm, 'mating') and hasattr(algorithm.mating, 'mutation'):
            stats['mutation_prob'] = algorithm.mating.mutation.prob
        if hasattr(algorithm, 'mating') and hasattr(algorithm.mating, 'crossover'):
            stats['crossover_prob'] = algorithm.mating.crossover.prob
            
        # Track Cumulative Discarded
        # We get the count from the current logger state before clearing
        current_discarded_count = sum(entry['total_count'] for entry in self.rule_logger.storage.values())
        self.cumulative_discarded += current_discarded_count
        stats['total_discarded_cumulative'] = self.cumulative_discarded
        stats['discarded_this_interval'] = current_discarded_count
            
        self.stats_history.append(stats)
        
        # Save Stats to CSV incrementally
        pd.DataFrame(self.stats_history).to_csv(self.stats_file, index=False)
        
        # 2. Save Population (All Individuals)
        self._save_population(pop, gen, 'population', algorithm, hv_value)
        
        # 3. Save Pareto Front (Non-dominated)
        # Pymoo's algorithm.opt holds the current best non-dominated solutions found so far
        opt = algorithm.opt
        if opt is not None and len(opt) > 0:
            self._save_population(opt, gen, 'pareto', algorithm, hv_value)
            
        # 4. Save Discarded Rules (Differential Snapshot)
        self._save_discarded(gen)
        
        # Clear logger for next interval (Option 3: Hybrid)
        self.rule_logger.storage = {}

    def _save_population(self, pop, gen, type_name, algorithm, hv_value=None):
        """
        Saves a population to CSV.
        Decodes individuals and includes metrics.
        """
        data = []
        real_F = -pop.get("F") # Convert back to real values
        X = pop.get("X")
        
        # Accessing metadata from problem
        problem = algorithm.problem if hasattr(algorithm, 'problem') else None
        
        for i, ind in enumerate(pop):
            row = {'generation': gen, 'id': i}
            
            if hv_value is not None:
                row['hypervolume'] = hv_value
            
            # Metrics
            for j, obj_name in enumerate(self.objectives):
                row[obj_name] = real_F[i, j]
                
            # Decode Rule
            # We need to create a temporary Individual to decode
            # This is slow but necessary for human-readable logs
            if problem:
                temp_ind = Individual(problem.metadata)
                temp_ind.X = X[i]
                ant_str, con_str = temp_ind.decode_parts()
                row['antecedent'] = ant_str
                row['consequent'] = con_str
                row['rule'] = f"{ant_str} => {con_str}"
                
                # Save raw genes
                row['encoded_rule'] = json.dumps(X[i].tolist())
            
            data.append(row)
            
        df = pd.DataFrame(data)
        filename = os.path.join(self.dirs['populations' if type_name == 'population' else 'pareto'], 
                                f'{type_name}_gen_{gen}.csv')
        df.to_csv(filename, index=False)

    def _save_discarded(self, gen):
        """
        Flushes the DiscardedRulesLogger to disk.
        """
        # The logger is passed in __init__
        # We access its storage
        storage = self.rule_logger.storage
        
        if not storage:
            return
            
        data = []
        for key, entry in storage.items():
            row = {
                'rule': entry['rule_decoded'],
                'total_count': entry['total_count'],
                'reasons': json.dumps(entry['reasons'])
            }
            # Add metrics if available (from the last discarded instance of this rule)
            if entry.get('metrics'):
                for m, v in entry['metrics'].items():
                    row[f"metric_{m}"] = v
            
            data.append(row)
            
        df = pd.DataFrame(data)
        filename = os.path.join(self.dirs['discarded'], f'discarded_gen_{gen}.csv')
        df.to_csv(filename, index=False)
        
        # Logger is cleared in notify() for Option 3 (Hybrid)
