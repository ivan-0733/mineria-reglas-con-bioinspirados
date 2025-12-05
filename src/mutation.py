import logging
import numpy as np
from pymoo.core.mutation import Mutation
from src.individual import Individual

class ARMMutation(Mutation):
    """
    Custom Mutation operator for Association Rule Mining (Diploid).
    Operations:
    - Extension: Activate a gene (Role 0 -> 1/2).
    - Contraction: Deactivate a gene (Role 1/2 -> 0).
    - Replacement: Change value of active gene.
    
    Includes repair mechanism and validation. Supports extra retries for duplicate collisions
    via duplicate_attempts without consuming validation attempts.
    """
    def __init__(self, metadata, validator, logger, prob=0.1, attempts=5, duplicate_attempts=0, active_ops=None):
        super().__init__()
        self.metadata = metadata
        self.validator = validator
        self.logger = logger
        self.prob = prob
        self.attempts = attempts
        self.duplicate_attempts = duplicate_attempts
        
        # Define available operations
        self.available_ops = ['extension', 'contraction', 'replacement']
        
        # Set active operations (default to all if None)
        if active_ops is None:
            self.active_ops = self.available_ops
        else:
            # Validate provided ops
            self.active_ops = [op for op in active_ops if op in self.available_ops]
            if not self.active_ops:
                raise ValueError("At least one valid mutation operator must be active.")
        
        # Helper to know cardinality
        self.dummy_ind = Individual(metadata)
        self.variables_info = self.dummy_ind.variables_info
        self.num_genes = self.dummy_ind.num_genes
        self.log = logging.getLogger(__name__)

    def _do(self, problem, X, **kwargs):
        # X shape: (n_individuals, n_var)
        n_ind, n_var = X.shape
        n_genes = n_var // 2
        
        # Reshape to (n_ind, 2, n_genes)
        Y = X.reshape(n_ind, 2, n_genes).copy()
        
        # Track unique signatures in this batch to avoid duplicates within offspring
        batch_signatures = set()
        for i in range(n_ind):
            ant, con = self._get_rule_items(Y[i])
            batch_signatures.add((frozenset(ant), frozenset(con)))
        
        for i in range(n_ind):
            if np.random.random() < self.prob:
                original = Y[i].copy()
                success = False

                validation_left = self.attempts
                duplicate_left = self.duplicate_attempts

                while (validation_left > 0) or (duplicate_left > 0):
                    mutant = original.copy()
                    
                    # Select Operation from active set
                    op = np.random.choice(self.active_ops)
                    
                    if op == 'extension':
                        self._apply_extension(mutant)
                    elif op == 'contraction':
                        self._apply_contraction(mutant)
                    elif op == 'replacement':
                        self._apply_replacement(mutant)
                        
                    # Repair structural damage (empty ant/con)
                    self._repair_structure(mutant)
                    
                    # Extract structure
                    ant, con = self._get_rule_items(mutant)
                    
                    # Check Duplicates in Batch
                    sig = (frozenset(ant), frozenset(con))
                    if sig in batch_signatures:
                        if duplicate_left > 0:
                            duplicate_left -= 1
                            continue
                        else:
                            validation_left -= 1
                            continue

                    # Validate
                    is_valid, reason, metrics = self.validator.validate(ant, con)
                    
                    if is_valid:
                        Y[i] = mutant
                        batch_signatures.add(sig) # Add new signature
                        success = True
                        break
                    else:
                        validation_left -= 1
                        # Log failure
                        temp_ind = Individual(self.metadata)
                        # temp_ind.X expects 1D now, but mutant is 2D
                        temp_ind.X = mutant.flatten()
                        self.logger.log(temp_ind, f"mutation_fail_{op}:{reason}", metrics)
                
                if not success:
                    # If all attempts fail, revert to original
                    Y[i] = original
                    self.log.warning(
                        "Mutation failed after attempts | ind=%s | ops=%s | validation_attempts=%s | duplicate_attempts=%s",
                        i,
                        self.active_ops,
                        self.attempts,
                        self.duplicate_attempts
                    )
                    
        return Y.reshape(n_ind, n_var)

    def _apply_extension(self, genome):
        # Find inactive genes (Role 0)
        inactive_indices = np.where(genome[0] == 0)[0]
        if len(inactive_indices) > 0:
            idx = np.random.choice(inactive_indices)
            # Assign Role 1 or 2
            genome[0, idx] = np.random.choice([1, 2])
            # Assign Value
            card = self.variables_info[idx]['cardinality']
            genome[1, idx] = np.random.randint(0, card)

    def _apply_contraction(self, genome):
        # Find active genes (Role != 0)
        active_indices = np.where(genome[0] != 0)[0]
        if len(active_indices) > 0:
            idx = np.random.choice(active_indices)
            genome[0, idx] = 0
            genome[1, idx] = 0

    def _apply_replacement(self, genome):
        # Find active genes
        active_indices = np.where(genome[0] != 0)[0]
        if len(active_indices) > 0:
            idx = np.random.choice(active_indices)
            card = self.variables_info[idx]['cardinality']
            if card > 1:
                current_val = genome[1, idx]
                # Pick new value different from current
                possible_vals = list(range(card))
                if current_val in possible_vals:
                    possible_vals.remove(current_val)
                if possible_vals:
                    genome[1, idx] = np.random.choice(possible_vals)

    def _repair_structure(self, genome):
        # Check if Antecedent (1) or Consequent (2) are empty
        roles = genome[0]
        has_ant = np.any(roles == 1)
        has_con = np.any(roles == 2)
        
        if not has_ant:
            # Force extension on a random available gene (or flip a consequent if no space)
            zeros = np.where(roles == 0)[0]
            if len(zeros) > 0:
                idx = np.random.choice(zeros)
                genome[0, idx] = 1
                card = self.variables_info[idx]['cardinality']
                genome[1, idx] = np.random.randint(0, card)
            else:
                # Flip a consequent to antecedent
                cons = np.where(roles == 2)[0]
                if len(cons) > 0:
                    idx = np.random.choice(cons)
                    genome[0, idx] = 1

        if not has_con:
            # Similar logic for consequent
            roles = genome[0] # Refresh
            zeros = np.where(roles == 0)[0]
            if len(zeros) > 0:
                idx = np.random.choice(zeros)
                genome[0, idx] = 2
                card = self.variables_info[idx]['cardinality']
                genome[1, idx] = np.random.randint(0, card)
            else:
                # Flip an antecedent to consequent
                ants = np.where(roles == 1)[0]
                if len(ants) > 0:
                    idx = np.random.choice(ants)
                    genome[0, idx] = 2

    def _get_rule_items(self, genome):
        # Helper to extract ant/con from genome numpy array
        antecedent = []
        consequent = []
        for i in range(self.num_genes):
            role = genome[0, i]
            val_idx = genome[1, i]
            if role == 1:
                antecedent.append((i, val_idx))
            elif role == 2:
                consequent.append((i, val_idx))
        return antecedent, consequent
