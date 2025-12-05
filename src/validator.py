import numpy as np

class ARMValidator:
    """
    Validates Association Rules based on:
    1. Structural Integrity (Disjoint, Non-empty)
    2. User-defined Constraints (Business Logic, Triviality)
    3. Metric Thresholds (Support, Confidence, etc.)
    """
    def __init__(self, config, metrics_engine, metadata):
        self.config = config
        self.metrics = metrics_engine
        self.metadata = metadata
        
        # Parse constraints from config
        self.constraints = config.get('constraints', {})
        self.validity_rules = self.constraints.get('rule_validity', {})
        self.thresholds = self.constraints.get('metric_thresholds', {})
        self.exclusions = self.constraints.get('exclusions', {})

        # Alias map to keep configs backward-compatible (e.g., "kappa" -> "k_measure")
        self.alias_map = {
            'kappa': 'k_measure'
        }
        
        # Cache variable names for fast lookup
        self.var_names = self._get_variable_names()

    def _get_variable_names(self):
        # Reconstruct the mapping from index to name used in the individual
        order = list(self.metadata.get('feature_order', []))
        target_name = self.metadata.get('target_variable')
        if isinstance(target_name, dict):
            target_name = target_name.get('name')
            
        if target_name and target_name not in order:
            order.append(target_name)
        return order

    def validate(self, antecedent, consequent):
        """
        Main validation pipeline.
        
        Args:
            antecedent: list of (var_idx, val_idx)
            consequent: list of (var_idx, val_idx)
            
        Returns:
            is_valid (bool): True if rule passes all checks
            reason (str): Reason for rejection (None if valid)
            metrics (dict): Calculated metrics (only if valid up to metric check)
        """
        
        # 1. Structural Validation (Fastest)
        if not self._check_structure(antecedent, consequent):
            return False, "structural_error", {}

        # 2. User/Business Constraints (Fast)
        if not self._check_user_constraints(antecedent, consequent):
            return False, "user_exclusion", {}

        # 3. Metric Validation (Slowest - requires probability calc)
        # We calculate ALL selected objectives here because we need to check thresholds
        # and also check for indeterminacy errors returned by the metrics engine.
        # Normalize metric names from config (thresholds may use aliases)
        objectives_to_check = [self.alias_map.get(k, k) for k in self.thresholds.keys()]
        
        # Also include objectives needed for optimization even if they don't have thresholds
        # (Though usually we filter by what we optimize)
        
        metric_values, errors = self.metrics.get_metrics(antecedent, consequent, objectives_to_check)
        
        # 3a. Check for Calculation Errors (Indeterminacy/Infinity)
        if errors:
            # Return the first error found
            first_error_metric = list(errors.keys())[0]
            return False, f"math_error:{errors[first_error_metric]}", {}

        # 3b. Check Thresholds
        metrics_dict = dict(zip(objectives_to_check, metric_values))
        
        for metric, value in metrics_dict.items():
            if value is None:
                 return False, f"undefined_{metric}", {}
                 
            # Pull thresholds using canonical name but fallback to original alias key if present
            limits = self.thresholds.get(metric, {})
            if not limits:
                for alias, canonical in self.alias_map.items():
                    if canonical == metric and alias in self.thresholds:
                        limits = self.thresholds[alias]
                        break
            if 'min' in limits and value < limits['min']:
                return False, f"below_min_{metric}", metrics_dict
            if 'max' in limits and value > limits['max']:
                return False, f"above_max_{metric}", metrics_dict

        return True, None, metrics_dict

    def _check_structure(self, antecedent, consequent):
        """
        Checks:
        - Non-empty X and Y
        - Disjoint X and Y (Intersection is empty)
        - Max items in X and Y (Complexity control)
        """
        # Non-empty
        if not antecedent or not consequent:
            return False
            
        # Check item counts against validity_rules
        min_ant = self.validity_rules.get('min_antecedent_items', 1)
        max_ant = self.validity_rules.get('max_antecedent_items', float('inf'))
        min_con = self.validity_rules.get('min_consequent_items', 1)
        max_con = self.validity_rules.get('max_consequent_items', float('inf'))

        if not (min_ant <= len(antecedent) <= max_ant):
            return False
        if not (min_con <= len(consequent) <= max_con):
            return False

        # Disjoint (Check variable indices)
        # antecedent is list of (var_idx, val_idx)
        vars_X = set(x[0] for x in antecedent)
        vars_Y = set(y[0] for y in consequent)
        
        if not vars_X.isdisjoint(vars_Y):
            return False
            
        return True

    def _check_user_constraints(self, antecedent, consequent):
        """
        Checks against 'exclusions' in config.
        Supported exclusions:
        - 'fixed_consequents': Variables that cannot be in consequent (e.g., 'race')
        - 'forbidden_pairs': Pairs of variables that cannot coexist in a rule
        """
        # Map indices to names for config comparison
        vars_X_names = set(self.var_names[x[0]] for x in antecedent)
        vars_Y_names = set(self.var_names[y[0]] for y in consequent)
        
        # 1. Fixed Consequents (Immutable variables shouldn't be predicted)
        # Example: You can't cause "Race" or "Sex"
        forbidden_targets = set(self.exclusions.get('fixed_consequents', []))
        if not vars_Y_names.isdisjoint(forbidden_targets):
            return False

        # 2. Forbidden Pairs (Logic check)
        # Example: "Pregnant" and "Male" should not appear in the same rule (X or Y)
        forbidden_pairs = self.exclusions.get('forbidden_pairs', [])
        all_vars = vars_X_names.union(vars_Y_names)
        
        for pair in forbidden_pairs:
            if set(pair).issubset(all_vars):
                return False

        return True
