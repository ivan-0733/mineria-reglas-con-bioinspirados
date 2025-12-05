import pandas as pd
import os
import json

class DiscardedRulesLogger:
    """
    Manages the storage and aggregation of invalid/discarded rules during the evolutionary process.
    Aggregates duplicates to save memory and provide statistical insights.
    """
    def __init__(self):
        # Storage structure:
        # Key: decoded_rule_string
        # Value: dict containing details and reason counts
        self.storage = {}

    def log(self, individual, reason, metrics=None):
        """
        Logs a discarded individual.
        
        Args:
            individual: The Individual object (must have decode() and X attributes)
            reason (str): The reason for rejection
            metrics (dict, optional): Calculated metrics at the time of rejection
        """
        try:
            decoded_rule = individual.decode()
        except Exception:
            decoded_rule = "Error decoding rule"

        # Use decoded rule as key to aggregate same rules with potentially different reasons
        key = decoded_rule
        
        if key not in self.storage:
            # Initialize entry
            self.storage[key] = {
                'rule_decoded': decoded_rule,
                'rule_encoded': json.dumps(individual.X.tolist()), # Store genes as string
                'metrics': metrics if metrics else {},
                'reasons': {},
                'total_count': 0
            }
        
        # Update counts
        entry = self.storage[key]
        entry['total_count'] += 1
        
        # Update specific reason count
        if reason not in entry['reasons']:
            entry['reasons'][reason] = 0
        entry['reasons'][reason] += 1
        
        # Update metrics if we have them now but didn't before (edge case)
        if metrics and not entry['metrics']:
             entry['metrics'] = metrics

    def save(self, output_dir, filename="discarded_rules_stats.csv"):
        """
        Saves the aggregated stats to a CSV file.
        Creates dynamic columns for each reason encountered.
        """
        if not self.storage:
            print("No discarded rules to save.")
            return
            
        print(f"Saving {len(self.storage)} unique discarded rules (aggregated)...")
        
        data = []
        all_reasons = set()
        
        # First pass: collect all unique reasons to create columns
        for entry in self.storage.values():
            all_reasons.update(entry['reasons'].keys())
            
        # Second pass: flatten data
        for entry in self.storage.values():
            row = {
                'rule_decoded': entry['rule_decoded'],
                'rule_encoded': entry['rule_encoded'],
                'total_count': entry['total_count']
            }
            
            # Add metrics columns
            metrics = entry.get('metrics', {})
            if metrics:
                for k, v in metrics.items():
                    # Handle None values in metrics
                    row[f"metric_{k}"] = v if v is not None else "NaN"
            
            # Add reason columns (fill with 0 if not present for this rule)
            for r in all_reasons:
                row[f"reason_{r}"] = entry['reasons'].get(r, 0)
                
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Sort by total_count descending to show most frequent errors first
        if 'total_count' in df.columns:
            df = df.sort_values('total_count', ascending=False)
            
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False, encoding='utf-8')
        print(f"✅ Discarded rules report saved to: {path}")

    def get_summary(self):
        """Returns a simple summary of discard reasons across all rules."""
        summary = {}
        for entry in self.storage.values():
            for r, count in entry['reasons'].items():
                summary[r] = summary.get(r, 0) + count
        return summary
