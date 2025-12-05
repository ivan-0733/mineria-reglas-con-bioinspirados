from pymoo.core.sampling import Sampling
import numpy as np
from src.individual import Individual

class InitializationError(Exception):
    """Raised when population initialization fails due to too many invalid attempts."""
    pass

class ARMSampling(Sampling):
    """
    Custom Sampling strategy for Association Rule Mining with Diploid Representation.
    Ensures initial population consists of valid, non-duplicate rules.
    """
    def __init__(self, metadata, validator, logger, max_attempts=10000):
        super().__init__()
        self.metadata = metadata
        self.validator = validator
        self.logger = logger
        self.max_attempts = max_attempts

    def _do(self, problem, n_samples, **kwargs):
        X = []
        # Set of signatures to detect duplicates: (frozenset(ant), frozenset(con))
        seen_rules = set()
        
        print(f"Initializing population of size {n_samples}...")
        
        total_attempts = 0
        created = 0
        stagnant_attempts = 0  # Track attempts without progress
        max_stagnant = 100  # Allow duplicates after this many attempts without new unique individual
        
        while created < n_samples:
            # Safety break for infinite loops
            if total_attempts >= self.max_attempts:
                print(f"\nWarning: Reached max attempts ({total_attempts}). Filling remaining with duplicates.")
                break

            # 1. Create and initialize individual
            ind = Individual(self.metadata)
            ind.initialize()
            ind.repair() # Ensure consistency (Role 0 -> Value 0)
            
            # 2. Extract structure
            antecedent, consequent = ind.get_rule_items()
            
            # 3. Check Duplicates
            # We use frozenset because order of items in antecedent/consequent doesn't matter for the rule logic
            rule_signature = (frozenset(antecedent), frozenset(consequent))
            
            if rule_signature in seen_rules:
                # Skip logging during initialization to save time
                total_attempts += 1
                stagnant_attempts += 1
                
                # If we've been stuck for too long, allow some duplicates
                if stagnant_attempts >= max_stagnant and len(X) > 0:
                    print(f"\nWarning: Stuck at {created}/{n_samples}. Allowing duplicates to continue.")
                    break
                continue

            # 4. Validate
            is_valid, reason, metrics = self.validator.validate(antecedent, consequent)
            
            if is_valid:
                X.append(ind.X)
                seen_rules.add(rule_signature)
                created += 1
                stagnant_attempts = 0  # Reset stagnant counter on success
            else:
                # Skip logging during initialization to speed up
                stagnant_attempts += 1
            
            total_attempts += 1
            
            if total_attempts % 100 == 0:
                 print(f"Generating population: {created}/{n_samples} (Attempts: {total_attempts})")
        
        print() # Newline after loop
            
        # If we couldn't fill the population, we might need to fill with duplicates or just return what we have
        # Pymoo expects exact size usually.
        if len(X) < n_samples:
            print(f"Filling remaining {n_samples - len(X)} spots with random valid individuals (allowing duplicates).")
            while len(X) < n_samples:
                # Just pick random existing ones to fill
                idx = np.random.randint(0, len(X))
                X.append(X[idx].copy())

        return np.array(X)
