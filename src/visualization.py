import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.plotting import parallel_coordinates

class VisualizationManager:
    """
    Manages the generation of plots for the MOEA/D ARM algorithm.
    """
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set style
        sns.set_theme(style="whitegrid")
        self.objectives = config['objectives']['selected']

    def generate_all(self):
        print("Generating visualizations...")
        self._plot_metrics_evolution()
        self._plot_hypervolume_evolution()
        self._plot_discarded_stats()
        self._plot_pareto_fronts()
        print(f"Plots saved to {self.plots_dir}")

    def _plot_metrics_evolution(self):
        stats_path = os.path.join(self.output_dir, 'stats', 'evolution_stats.csv')
        if not os.path.exists(stats_path):
            return
            
        df = pd.read_csv(stats_path)
        
        # Plot each objective
        for obj in self.objectives:
            plt.figure(figsize=(10, 6))
            
            # Mean line
            sns.lineplot(data=df, x='generation', y=f'{obj}_mean', label='Mean', color='blue')
            
            # Min-Max Area
            plt.fill_between(df['generation'], df[f'{obj}_min'], df[f'{obj}_max'], alpha=0.2, color='blue', label='Min-Max Range')
            
            plt.title(f'Evolution of {obj}')
            plt.xlabel('Generation')
            plt.ylabel(obj)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'evolution_{obj}.png'))
            plt.close()

    def _plot_hypervolume_evolution(self):
        stats_path = os.path.join(self.output_dir, 'stats', 'evolution_stats.csv')
        if not os.path.exists(stats_path):
            return
            
        df = pd.read_csv(stats_path)
        if 'hypervolume' not in df.columns:
            return

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='generation', y='hypervolume', color='green', marker='o')
        plt.title('Hypervolume Evolution')
        plt.xlabel('Generation')
        plt.ylabel('Hypervolume')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'evolution_hypervolume.png'))
        plt.close()

    def _plot_discarded_stats(self):
        # Load all discarded logs
        discarded_dir = os.path.join(self.output_dir, 'discarded')
        if not os.path.exists(discarded_dir):
            return
            
        files = sorted([f for f in os.listdir(discarded_dir) if f.endswith('.csv') and f.startswith('discarded_gen_')])
        if not files:
            return
            
        # Aggregate data
        data = []
        for f in files:
            try:
                gen = int(f.split('_')[-1].replace('.csv', ''))
            except ValueError:
                continue
                
            df = pd.read_csv(os.path.join(discarded_dir, f))
            
            # Parse reasons (JSON)
            # We want to count total discards per reason type for this generation
            reason_counts = {}
            for _, row in df.iterrows():
                count = row['total_count']
                # reasons is a stringified dict like "{'reason1': count, ...}"
                # But wait, the logger saves reasons as a list of strings in the key? 
                # Let's check DiscardedRulesLogger. 
                # Actually, callback saves: 'reasons': json.dumps(entry['reasons'])
                # entry['reasons'] is a dict {reason: count}
                import json
                try:
                    reasons_dict = json.loads(row['reasons'])
                    for r, c in reasons_dict.items():
                        reason_counts[r] = reason_counts.get(r, 0) + c
                except:
                    pass
            
            for r, c in reason_counts.items():
                data.append({'generation': gen, 'reason': r, 'count': c})
                
        if not data:
            return
            
        df_all = pd.DataFrame(data)
        
        # Stacked Bar Chart
        pivot_df = df_all.pivot_table(index='generation', columns='reason', values='count', aggfunc='sum', fill_value=0)
        
        pivot_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
        plt.title('Discarded Rules by Reason per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Count')
        plt.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'discarded_stats.png'))
        plt.close()

    def _plot_pareto_fronts(self):
        pareto_dir = os.path.join(self.output_dir, 'pareto')
        if not os.path.exists(pareto_dir):
            return
            
        files = sorted([f for f in os.listdir(pareto_dir) if f.endswith('.csv')])
        
        for f in files:
            gen = f.split('_')[-1].replace('.csv', '')
            df = pd.read_csv(os.path.join(pareto_dir, f))
            
            if len(self.objectives) == 3:
                self._plot_3d_scatter(df, gen)
            elif len(self.objectives) > 3:
                self._plot_parallel_coordinates(df, gen)
            else:
                # 2D Plot
                self._plot_2d_scatter(df, gen)

    def _plot_3d_scatter(self, df, gen):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x_col, y_col, z_col = self.objectives[:3]
        
        sc = ax.scatter(df[x_col], df[y_col], df[z_col], c=df[z_col], cmap='viridis', s=50, alpha=0.8)
        
        # Ideal Point (Assuming minimization of negative values -> Maximize real values)
        # Ideal is (1, 1, 1) for normalized metrics
        ax.scatter([1], [1], [1], color='red', marker='*', s=200, label='Ideal Point')
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f'Pareto Front - Gen {gen}')
        plt.colorbar(sc, label=z_col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'pareto_3d_gen_{gen}.png'))
        plt.close()

    def _plot_parallel_coordinates(self, df, gen):
        plt.figure(figsize=(12, 6))
        
        # Normalize data for better visualization if needed, but metrics are usually [0,1] or [-1,1]
        # Let's plot raw values
        cols = self.objectives
        
        # Add a dummy class column for coloring if not present
        plot_df = df[cols].copy()
        plot_df['id'] = df.index # Use index as class to color lines differently
        
        # Use pandas parallel_coordinates
        # But pandas parallel_coordinates needs a class column for color.
        # Let's use a custom implementation with matplotlib for better control or just pandas
        
        # Better: Parallel Coordinates with color based on Hypervolume or first objective
        # Since we don't have a categorical class, we can't easily use pandas parallel_coordinates for gradient color.
        # We'll use a loop.
        
        ax = plt.gca()
        for i, row in plot_df.iterrows():
            # Color by first objective
            val = row[cols[0]]
            # Normalize val for color
            color = plt.cm.viridis(val) # Assuming val in [0,1]
            
            ys = row[cols].values
            xs = range(len(cols))
            ax.plot(xs, ys, color=color, alpha=0.5)
            
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols)
        plt.title(f'Parallel Coordinates - Gen {gen}')
        plt.xlabel('Objectives')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'pareto_parallel_gen_{gen}.png'))
        plt.close()

    def _plot_2d_scatter(self, df, gen):
        plt.figure(figsize=(8, 6))
        x_col, y_col = self.objectives[:2]
        
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=y_col, palette='viridis', s=100)
        plt.scatter([1], [1], color='red', marker='*', s=200, label='Ideal Point')
        
        plt.title(f'Pareto Front - Gen {gen}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'pareto_2d_gen_{gen}.png'))
        plt.close()
