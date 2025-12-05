import numpy as np
import pandas as pd

class ARMMetrics:
    """
    Calculates objectives and metrics for Association Rules.
    Uses pre-calculated supports for marginal probabilities (P(A), P(C))
    and the dataset (sample or full) for joint probabilities (P(A and C)).
    """
    def __init__(self, dataframe, supports_dict, metadata):
        self.df = dataframe
        self.supports = supports_dict
        self.metadata = metadata
        self.total_rows = len(dataframe)

        # Alias map to keep configs backward-compatible (e.g., "kappa" -> "k_measure")
        self.alias_map = {
            'kappa': 'k_measure'
        }
        
        # Parse variable order to map indices to names
        self.var_names = self._get_variable_order()
        
        # Cache to store calculated metrics for rules to avoid re-computation
        # Key: (frozenset(antecedent), frozenset(consequent))
        # Value: dict of metrics
        self._cache = {}

    def _get_variable_order(self):
        order = list(self.metadata.get('feature_order', []))
        target_name = self.metadata.get('target_variable')
        if isinstance(target_name, dict):
            target_name = target_name.get('name')
            
        if target_name and target_name not in order:
            order.append(target_name)
        return order

    def _get_probability(self, items):
        """
        Calculates P(items).
        items: list of (var_idx, val_idx)
        """
        if not items:
            return 0.0
            
        # Optimization: Single item -> Use supports.json
        if len(items) == 1:
            var_idx, val_idx = items[0]
            var_name = self.var_names[var_idx]
            # Supports keys are strings
            try:
                return self.supports['variables'][var_name][str(val_idx)]
            except KeyError:
                return 0.0
        
        # Multiple items -> Query DataFrame
        # Construct query mask
        mask = np.ones(self.total_rows, dtype=bool)
        for var_idx, val_idx in items:
            var_name = self.var_names[var_idx]
            # Assuming df has integer values matching val_idx
            mask &= (self.df[var_name] == val_idx)
            
        count = mask.sum()
        return count / self.total_rows

    def get_metrics(self, antecedent, consequent, objectives):
        """
        Returns a list of values for the selected objectives.
        antecedent: list of (variable_index, value_index) tuples
        consequent: list of (variable_index, value_index) tuples
        
        Returns:
            list: Metric values. If a metric is invalid, returns None for that position.
            dict: Error details if any metric failed (e.g., {'max_conf': 'zero_antecedent_support'})
        """
        # 1. Check Cache
        # Normalize objective names to canonical keys
        canonical_objectives = [self.alias_map.get(obj, obj) for obj in objectives]

        rule_key = (frozenset(antecedent), frozenset(consequent))
        if rule_key in self._cache:
            cached = self._cache[rule_key]
            # Check if all objectives are in cache using canonical names
            if all(obj in cached for obj in canonical_objectives):
                # Return cached values mapped back to original order
                return [cached[self.alias_map.get(obj, obj)] for obj in objectives], {}

        # 2. Calculate Basic Probabilities
        P_X = self._get_probability(antecedent)
        P_Y = self._get_probability(consequent)
        
        # Union items for P(X and Y)
        union_items = list(set(antecedent + consequent))
        P_XY = self._get_probability(union_items)
        
        # Derived Probabilities
        P_not_X = 1.0 - P_X
        P_not_Y = 1.0 - P_Y
        P_not_X_not_Y = 1.0 - (P_X + P_Y - P_XY)
        P_not_X_not_Y = max(0.0, min(1.0, P_not_X_not_Y))

        metrics = {}
        errors = {}

        # --- Pre-validation Checks ---
        # Identify conditions that cause indeterminacy
        is_X_zero = (P_X == 0)
        is_Y_zero = (P_Y == 0)
        is_X_one = (P_X == 1)
        is_Y_one = (P_Y == 1)
        
        # --- Calculate Requested Objectives ---
        
        # 0. Standard Support
        # P(X n Y)
        if 'support' in canonical_objectives:
            metrics['support'] = P_XY

        # 0. Standard Confidence
        # P(Y|X)
        if 'confidence' in canonical_objectives:
            if is_X_zero:
                metrics['confidence'] = None
                errors['confidence'] = 'zero_antecedent_support'
            else:
                metrics['confidence'] = P_XY / P_X

        # 0. Lift
        # P(X n Y) / (P(X) * P(Y))
        if 'lift' in canonical_objectives:
            if is_X_zero or is_Y_zero:
                metrics['lift'] = None
                errors['lift'] = 'zero_marginal_support'
            else:
                metrics['lift'] = P_XY / (P_X * P_Y)

        # 1. Casual Support
        # casual-supp = P(X n Y) + P(not X n not Y)
        if 'casual_support' in canonical_objectives:
            metrics['casual_support'] = P_XY + P_not_X_not_Y

        # 2. Casual Confidence
        # casual-conf = 0.5 * [P(Y|X) + P(not Y | not X)]
        if 'casual_confidence' in canonical_objectives:
            if is_X_zero:
                metrics['casual_confidence'] = None
                errors['casual_confidence'] = 'zero_antecedent_support'
            elif P_not_X == 0: # Equivalent to is_X_one
                metrics['casual_confidence'] = None
                errors['casual_confidence'] = 'full_antecedent_support'
            else:
                conf_X_Y = P_XY / P_X
                conf_not_X_not_Y = P_not_X_not_Y / P_not_X
                metrics['casual_confidence'] = 0.5 * (conf_X_Y + conf_not_X_not_Y)

        # 3. Max Confidence
        # maxConf = max(P(Y|X), P(X|Y))
        if 'max_conf' in canonical_objectives:
            if is_X_zero:
                metrics['max_conf'] = None
                errors['max_conf'] = 'zero_antecedent_support'
            elif is_Y_zero:
                metrics['max_conf'] = None
                errors['max_conf'] = 'zero_consequent_support'
            else:
                conf_X_Y = P_XY / P_X
                conf_Y_X = P_XY / P_Y
                metrics['max_conf'] = max(conf_X_Y, conf_Y_X)

        # 4. Jaccard
        # P(X n Y) / (P(X) + P(Y) - P(X n Y))
        if 'jaccard' in canonical_objectives:
            denom = P_X + P_Y - P_XY
            if denom == 0:
                metrics['jaccard'] = None
                errors['jaccard'] = 'zero_union_support'
            else:
                metrics['jaccard'] = P_XY / denom

        # 5. Cosine
        # P(X n Y) / sqrt(P(X) * P(Y))
        if 'cosine' in canonical_objectives:
            if is_X_zero or is_Y_zero:
                metrics['cosine'] = None
                errors['cosine'] = 'zero_support_X_or_Y'
            else:
                denom = np.sqrt(P_X * P_Y)
                metrics['cosine'] = P_XY / denom

        # 6. Phi Correlation
        # (P(XY) - P(X)P(Y)) / sqrt(P(X)(1-P(X))P(Y)(1-P(Y)))
        if 'phi' in canonical_objectives:
            if is_X_zero or is_X_one or is_Y_zero or is_Y_one:
                metrics['phi'] = None
                errors['phi'] = 'zero_variance_X_or_Y'
            else:
                num = P_XY - (P_X * P_Y)
                denom = np.sqrt(P_X * P_not_X * P_Y * P_not_Y)
                metrics['phi'] = num / denom

        # 7. K-measure (Kappa)
        # (P(XY) + P(notX notY) - P(X)P(Y) - P(notX)P(notY)) / (1 - P(X)P(Y) - P(notX)P(notY))
        if 'k_measure' in canonical_objectives:
            chance_agreement = (P_X * P_Y) + (P_not_X * P_not_Y)
            if chance_agreement == 1.0: # Deterministic case
                metrics['k_measure'] = None
                errors['k_measure'] = 'deterministic_agreement'
            else:
                obs_agreement = P_XY + P_not_X_not_Y
                num = obs_agreement - chance_agreement
                denom = 1.0 - chance_agreement
                metrics['k_measure'] = num / denom

        # 3. Store in Cache (update existing dict if any)
        if rule_key not in self._cache:
            self._cache[rule_key] = {}
        self._cache[rule_key].update(metrics)

        # Map results back to the original objective names
        values_in_original_order = [metrics.get(self.alias_map.get(obj, obj), None) for obj in objectives]

        # Remap error keys to the original names for clarity
        remapped_errors = {}
        for key, val in errors.items():
            original_key = key
            for alias, canonical in self.alias_map.items():
                if key == canonical:
                    original_key = alias
                    break
            remapped_errors[original_key] = val

        return values_in_original_order, remapped_errors

    def clear_cache(self):
        self._cache = {}
