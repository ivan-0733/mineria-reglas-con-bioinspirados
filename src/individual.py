import numpy as np
import random
from pymoo.core.individual import Individual as PymooIndividual

class Individual(PymooIndividual):
    """
    Represents an individual in the MOEA/D algorithm for association rule mining.
    Uses a diploid representation:
    - Chromosome 1: Role of the variable (0: None, 1: Antecedent, 2: Consequent)
    - Chromosome 2: Value of the variable (Integer index corresponding to the category)
    """
    def __init__(self, metadata=None, **kwargs):
        super().__init__(**kwargs)
        self.metadata = metadata
        if self.metadata:
            self.variables_info = self._parse_variables()
            self.num_genes = len(self.variables_info)
            # X will hold the diploid structure flattened: 
            # First num_genes are Roles, next num_genes are Values
            self.X = np.zeros(2 * self.num_genes, dtype=int)

    def _parse_variables(self):
        """
        Extracts variable information (name, cardinality, labels) from metadata.
        Uses the 'variables' dictionary and 'feature_order'.
        """
        if not self.metadata:
            return []

        variables_info = []
        
        # Get the order of features
        order = list(self.metadata.get('feature_order', []))
        
        # Add target variable if it exists and is not in the list
        target_name = self.metadata.get('target_variable')
        if target_name and target_name not in order:
            order.append(target_name)
            
        # Iterate through the ordered names and fetch info from 'variables'
        variables_dict = self.metadata.get('variables', {})
        
        for name in order:
            if name in variables_dict:
                info = variables_dict[name]
                if 'cardinality' in info:
                    variables_info.append({
                        'name': name,
                        'cardinality': info['cardinality'],
                        'labels': info.get('labels', []),
                        'encoding': info.get('encoding', {})
                    })
                else:
                    raise ValueError(f"Variable '{name}' lacks 'cardinality' information.")
        
        return variables_info

    def initialize(self):
        """
        Randomly initializes the individual.
        Ensures consistency: if Role is 0, Value is 0.
        """
        for i in range(self.num_genes):
            role_idx = i
            val_idx = self.num_genes + i
            
            # Randomly assign role: 0 (Ignore), 1 (Antecedent), 2 (Consequent)
            # Modified to favor sparsity (more 0s) to avoid zero-support rules in initialization
            # Probabilities: 60% Ignore, 20% Antecedent, 20% Consequent
            self.X[role_idx] = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])
            
            if self.X[role_idx] != 0:
                # Assign random value based on cardinality
                card = self.variables_info[i]['cardinality']
                self.X[val_idx] = random.randint(0, card - 1)
            else:
                self.X[val_idx] = 0

    def repair(self):
        """
        Repairs the individual to ensure consistency.
        Rule: If Role is 0, Value must be 0.
        Also ensures values are within valid cardinality range.
        """
        for i in range(self.num_genes):
            role_idx = i
            val_idx = self.num_genes + i
            
            if self.X[role_idx] == 0:
                self.X[val_idx] = 0
            else:
                card = self.variables_info[i]['cardinality']
                # Clamp value to valid range
                if self.X[val_idx] >= card:
                    self.X[val_idx] = card - 1
                if self.X[val_idx] < 0:
                    self.X[val_idx] = 0

    def get_rule_items(self):
        """
        Extracts the rule structure as lists of (variable_index, value_index).
        Returns:
            antecedent: list of (var_idx, val_idx)
            consequent: list of (var_idx, val_idx)
        """
        antecedent = []
        consequent = []
        
        for i in range(self.num_genes):
            role = self.X[i]
            val = self.X[self.num_genes + i]
            
            if role == 1: # Antecedent
                antecedent.append((i, val))
            elif role == 2: # Consequent
                consequent.append((i, val))
                
        return antecedent, consequent

    def decode_parts(self):
        """
        Decodes the individual into human-readable antecedent and consequent strings.
        Returns: (antecedent_str, consequent_str)
        """
        antecedent_items, consequent_items = self.get_rule_items()
        
        def format_items(items):
            parts = []
            for var_idx, val_idx in items:
                var_info = self.variables_info[var_idx]
                var_name = var_info['name']
                
                # Try to get label if available
                labels = var_info.get('labels', [])
                if labels and 0 <= val_idx < len(labels):
                    val_str = labels[val_idx]
                else:
                    val_str = str(val_idx)
                    
                parts.append(f"{var_name}={val_str}")
            return " ^ ".join(parts)

        ant_str = format_items(antecedent_items)
        con_str = format_items(consequent_items)
        
        if not ant_str: ant_str = "{}"
        if not con_str: con_str = "{}"
        
        return ant_str, con_str

    def decode(self):
        """
        Decodes the individual into a human-readable rule string.
        Format: Antecedent => Consequent
        """
        ant_str, con_str = self.decode_parts()
        return f"{ant_str} => {con_str}"

    def __repr__(self):
        return self.decode()
