import numpy as np
from pymoo.core.crossover import Crossover

class DiploidNPointCrossover(Crossover):
    """
    N-Point Crossover for Diploid Individuals.
    
    Logic:
    1. Selects N cut points randomly.
    2. Swaps segments (blocks of genes) between parents alternating at each cut point.
    3. Preserves the vertical structure (Role and Value columns stay together).
    """
    def __init__(self, n_points=2, prob=0.9, **kwargs):
        # 2 parents, 2 offspring
        super().__init__(2, 2, **kwargs)
        self.n_points = n_points
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        """
        Perform the crossover.
        X shape: (n_matings, n_parents, n_var)
        """
        n_matings, n_parents, n_var = X.shape
        n_genes = n_var // 2
        
        # Reshape to (n_matings, n_parents, 2, n_genes)
        # Assuming X is flattened [Role0...RoleN, Val0...ValN]
        X_reshaped = X.reshape(n_matings, n_parents, 2, n_genes)
        Y_reshaped = np.zeros_like(X_reshaped)
        
        for i in range(n_matings):
            parent1 = X_reshaped[i, 0]
            parent2 = X_reshaped[i, 1]
            
            # Select cut points
            # We need n_points between 1 and n_genes-1
            if n_genes > self.n_points:
                # Choose unique sorted cut points
                cut_points = np.sort(np.random.choice(np.arange(1, n_genes), self.n_points, replace=False))
                # Add start (0) and end (n_genes) for easier looping
                points = np.concatenate(([0], cut_points, [n_genes]))
            else:
                # Fallback if genes < points (just swap everything or 1 point)
                points = np.array([0, n_genes])
            
            # Create offspring
            off1 = np.zeros_like(parent1)
            off2 = np.zeros_like(parent2)
            
            # Fill segments
            swap = False
            for j in range(len(points) - 1):
                start, end = points[j], points[j+1]
                
                if not swap:
                    # Direct copy
                    off1[:, start:end] = parent1[:, start:end]
                    off2[:, start:end] = parent2[:, start:end]
                else:
                    # Swap copy
                    off1[:, start:end] = parent2[:, start:end]
                    off2[:, start:end] = parent1[:, start:end]
                
                # Toggle swap for next segment
                swap = not swap
                
            Y_reshaped[i, 0] = off1
            Y_reshaped[i, 1] = off2
            
        # Flatten back
        Y = Y_reshaped.reshape(n_matings, n_parents, n_var)
        # print(f"DEBUG: Crossover output shape: {Y.shape}")
        return Y
