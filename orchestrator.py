import os
import json
import logging
import pandas as pd
from src.MOEAD import MOEAD_ARM
from src.callback import ARMCallback
from src.individual import Individual
from pymoo.indicators.hv import HV
import numpy as np

class Orchestrator:
    """
    Orchestrates the execution of the MOEA/D ARM algorithm.
    Handles data loading, configuration setup, and execution management.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.data_context = self._load_data()
        
    def _load_config(self, path):
        print(f"Loading configuration from {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            exp_config = json.load(f)
            
        # Load base config
        base_config_path = os.path.join('config', 'general', 'base_config.json')
        if os.path.exists(base_config_path):
            print(f"Loading base configuration from {base_config_path}...")
            with open(base_config_path, 'r', encoding='utf-8') as f:
                base_config = json.load(f)
            
            # Merge dataset info from base to exp
            if 'dataset' in base_config:
                # If experiment config has dataset overrides, we might want to keep them
                # But usually base config has the paths.
                # Let's ensure 'dataset' key exists in exp_config
                if 'dataset' not in exp_config:
                    exp_config['dataset'] = {}
                
                # Update with base values (base provides defaults)
                for k, v in base_config['dataset'].items():
                    if k not in exp_config['dataset']:
                        exp_config['dataset'][k] = v
        else:
            print(f"Warning: Base config not found at {base_config_path}")
            
        return exp_config

    def _load_data(self):
        """
        Loads the necessary data files based on the configuration.
        Expects processed data to exist.
        """
        print("Loading data context...")
        
        dataset_config = self.config.get('dataset', {})
        if not dataset_config:
            raise ValueError("Dataset configuration missing. Ensure base_config.json is loaded correctly.")

        # Check if sample is used
        use_sample = self.config.get('use_sampling', False)
        
        if use_sample:
            print("Using sampled data...")
            csv_path = dataset_config.get('sampling_path')
            metadata_path = dataset_config.get('sample_metadata_path')
            supports_path = dataset_config.get('sample_supports_path')
        else:
            print("Using full dataset...")
            csv_path = dataset_config.get('processed_path')
            metadata_path = dataset_config.get('metadata_path')
            supports_path = dataset_config.get('supports_path')
            
        # Validate paths
        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        if not metadata_path or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not supports_path or not os.path.exists(supports_path):
            raise FileNotFoundError(f"Supports file not found: {supports_path}")
            
        df = pd.read_csv(csv_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        with open(supports_path, 'r', encoding='utf-8') as f:
            supports = json.load(f)
            
        return {
            'df': df,
            'metadata': metadata,
            'supports': supports
        }

    def run(self):
        # Setup output directory
        output_root = self.config['experiment'].get('output_root', 'results')
        exp_name = self.config['experiment']['name']
        base_exp_dir = os.path.join(output_root, exp_name)
        
        # Ensure base experiment directory exists
        if not os.path.exists(base_exp_dir):
            os.makedirs(base_exp_dir, exist_ok=True)

        # Find next available experiment folder (exp_001, exp_002, ...)
        counter = 1
        while True:
            folder_name = f"exp_{counter:03d}"
            exp_dir = os.path.join(base_exp_dir, folder_name)
            if not os.path.exists(exp_dir):
                break
            counter += 1
        
        os.makedirs(exp_dir, exist_ok=True)
        print(f"Experiment Output Directory: {exp_dir}")

        # Configure debug logging to file and console for this run
        log_file = os.path.join(exp_dir, 'debug_run.log')
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                handlers=[
                    logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
        else:
            # Ensure file handler is attached even if logging was configured elsewhere
            logging.getLogger().addHandler(logging.FileHandler(log_file, mode='w', encoding='utf-8'))
        logging.info("Logging initialized for experiment at %s", exp_dir)
        
        # Save used config for reproducibility
        with open(os.path.join(exp_dir, 'config_snapshot.json'), 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Initializing Algorithm (Output: {exp_dir})...")
        
        # Init Algorithm Wrapper
        moead_arm = MOEAD_ARM(self.config, self.data_context)
        
        # Init Callback
        callback = ARMCallback(self.config, moead_arm.logger, exp_dir)
        
        print("Starting Optimization...")
        # Run
        res = moead_arm.run(callback=callback)
        
        # Save Discarded Rules Log
        print("Saving discarded rules log...")
        moead_arm.logger.save(os.path.join(exp_dir, 'discarded'))
        
        # Save Final Pareto Front explicitly
        if res.opt is not None and len(res.opt) > 0:
            print("Saving final pareto front...")
            # Filter duplicates in Pareto Front based on X (Genotype)
            # MOEA/D can have multiple subproblems converging to the same solution
            unique_indices = []
            seen_X = set()
            X = res.opt.get("X")
            for i in range(len(res.opt)):
                # Convert row to tuple for hashing
                # X[i] is (2, n_features), flatten to make it hashable (tuple of ints)
                x_tuple = tuple(X[i].flatten())
                if x_tuple not in seen_X:
                    seen_X.add(x_tuple)
                    unique_indices.append(i)
            
            unique_pareto = res.opt[unique_indices]
            print(f"Unique solutions in Pareto: {len(unique_pareto)} (Original: {len(res.opt)})")
            
            self._save_population(unique_pareto, moead_arm.problem, os.path.join(exp_dir, 'final_pareto.csv'))
        
        # Consolidate historical Pareto front snapshots
        self._consolidate_history(exp_dir)
        
        print("Optimization Finished.")
        print(f"Best solutions found: {len(res.opt) if res.opt is not None else 0}")
        
        # Generate Visualizations
        try:
            from src.visualization import VisualizationManager
            viz = VisualizationManager(self.config, exp_dir)
            viz.generate_all()
        except Exception as e:
            print(f"Warning: Visualization generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return res

    def _save_population(self, pop, problem, filename):
        """
        Saves a population to CSV with detailed rule parts.
        """
        data = []
        real_F = -pop.get("F") # Convert back to real values
        X = pop.get("X")
        objectives = self.config['objectives']['selected']
        
        # Calculate Hypervolume for the set
        hv_value = 0.0
        try:
            # Reference point at 0 (since we minimize negative values [-1, 0])
            hv_indicator = HV(ref_point=np.zeros(len(objectives)))
            hv_value = hv_indicator(pop.get("F"))
        except Exception as e:
            print(f"Warning: Final HV calculation failed: {e}")

        for i, ind in enumerate(pop):
            row = {'id': i}
            
            # Hypervolume (repeated for all rows as it's a set metric)
            row['hypervolume'] = hv_value

            # Metrics
            for j, obj_name in enumerate(objectives):
                row[obj_name] = real_F[i, j]
                
            # Decode Rule
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
        df.to_csv(filename, index=False)
    
    def _consolidate_history(self, exp_dir):
        """
        Reads all pareto_gen_*.csv files, combines them, removes duplicates,
        and saves a consolidated history file.
        """
        pareto_dir = os.path.join(exp_dir, 'pareto')
        
        all_files = []
        if os.path.exists(pareto_dir):
            all_files = [f for f in os.listdir(pareto_dir) if f.startswith('pareto_gen_') and f.endswith('.csv')]

        dfs = []
        for f in all_files:
            path = os.path.join(pareto_dir, f)
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {f}: {e}")

        # Also include the final_pareto.csv if it exists
        final_pareto_path = os.path.join(exp_dir, 'final_pareto.csv')
        if os.path.exists(final_pareto_path):
            try:
                df = pd.read_csv(final_pareto_path)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read final_pareto.csv: {e}")

        if not dfs:
            return

        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates based on encoded_rule (genotype)
        # If encoded_rule is not present, use 'rule' string
        subset_cols = ['encoded_rule'] if 'encoded_rule' in combined_df.columns else ['rule']
        
        unique_df = combined_df.drop_duplicates(subset=subset_cols)
        
        # Save consolidated
        output_path = os.path.join(exp_dir, 'final_pareto_historical.csv')
        unique_df.to_csv(output_path, index=False)
        print(f"Consolidated historical Pareto front saved to {output_path}")
        print(f"Total unique solutions found across history: {len(unique_df)}")
