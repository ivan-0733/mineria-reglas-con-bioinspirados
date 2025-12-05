import json
import pandas as pd
import numpy as np
import os
import sys

class DatasetPreprocessor:
    def __init__(self, metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
    
    def preprocess(self, df, output_folder=None):
        """Aplica todas las transformaciones"""
        df = df.copy()

        # 0. Filtrar filas y eliminar columnas según metadata
        df = self._apply_filters(df)
        
        # 1. Rediscretizar variables (agrupar categorías)
        df = self._rediscretize_variables(df)

        # 2. Discretizar variables continuas (a etiquetas primero)
        df = self._discretize_continuous(df, to_labels=True)
        
        # 3. Decodificar one-hot encoding (a etiquetas)
        df = self._decode_one_hot(df, to_labels=True)
        
        # --- Generar reporte de distribución con etiquetas intuitivas ---
        if output_folder:
            self._generate_distribution_report(df, os.path.join(output_folder, "discretized_distribution.txt"))
        # ----------------------------------------------------------------

        # 4. Codificar variables categóricas a enteros
        df = self._encode_categorical(df)
        
        # 5. Auto-codificar variables restantes (texto)
        df = self._auto_encode_remaining(df)
        
        # 6. Enriquecer metadata de variable objetivo
        df = self._enrich_target_metadata(df)
        
        return df
    
    def _enrich_target_metadata(self, df):
        """Asegura que target_variable tenga estructura completa (type, mapping, cardinality)"""
        if 'target_variable' in self.metadata:
            target_info = self.metadata['target_variable']
            
            # Si es solo el nombre de la columna (string)
            if isinstance(target_info, str):
                target_col = target_info
                if target_col in df.columns:
                    print(f"Enriching metadata for target variable: {target_col}")
                    unique_vals = sorted(df[target_col].unique())
                    
                    # Crear mapping
                    mapping = {}
                    for val in unique_vals:
                        # Convertir numpy types a python types
                        key = str(val)
                        value = int(val) if isinstance(val, (int, np.integer)) else val
                        mapping[key] = value
                    
                    self.metadata['target_variable'] = {
                        "name": target_col,
                        "type": "nominal", # Default para clasificación
                        "mapping": mapping,
                        "cardinality": len(unique_vals)
                    }
                else:
                    print(f"Warning: Target variable {target_col} not found in dataframe.")
            
            # Si ya es un diccionario, verificamos si necesitamos actualizar algo (opcional)
            # Por ahora asumimos que si es dict ya está correcto o se deja como está
            
        return df

    def _export_metadata_structure(self):
        """Reestructura la metadata para que sea más accesible (variable-centric)"""
        new_meta = {
            "dataset_name": self.metadata.get("dataset_name", "unknown"),
            "variables": {},
            "feature_order": self.metadata.get("feature_order", []),
            "target_variable": self.metadata.get("target_variable", {}).get("name") if isinstance(self.metadata.get("target_variable"), dict) else self.metadata.get("target_variable"),
            "preprocessing": {
                "row_filters": self.metadata.get("row_filters", []),
                "drop_columns": self.metadata.get("drop_columns", [])
            }
        }

        # Helper to get or create var entry
        def get_var(name):
            if name not in new_meta["variables"]:
                new_meta["variables"][name] = {}
            return new_meta["variables"][name]

        # 1. Continuous to Categorical (Binning)
        if "continuous_to_categorical" in self.metadata:
            for col, config in self.metadata["continuous_to_categorical"].items():
                var = get_var(col)
                var["type"] = config.get("type", "ordinal")
                var["cardinality"] = config.get("cardinality")
                var["labels"] = config.get("labels")
                var["transformation"] = {
                    "source_type": "continuous",
                    "method": "binning",
                    "bins": config.get("bins"),
                    "right": config.get("right", False)
                }
                # Implicit encoding for binned vars (0..N)
                if config.get("labels"):
                    var["encoding"] = {label: i for i, label in enumerate(config["labels"])}

        # 2. One Hot to Decode
        if "one_hot_to_decode" in self.metadata:
            for col, config in self.metadata["one_hot_to_decode"].items():
                var = get_var(col)
                var["transformation"] = {
                    "source_type": "one_hot",
                    "columns": config.get("columns")
                }

        # 3. Rediscretization
        if "rediscretization" in self.metadata:
            for col, config in self.metadata["rediscretization"].items():
                var = get_var(col)
                var["transformation"] = {
                    "source_type": "nominal",
                    "method": "hierarchical_grouping",
                    "original_cardinality": config.get("original_cardinality"),
                    "strategy": config.get("grouping_strategy"),
                    "groups": config.get("mapping")
                }

        # 4. Categorical Encoding (The source of truth for final integer mapping)
        if "categorical_encoding" in self.metadata:
            for col, config in self.metadata["categorical_encoding"].items():
                var = get_var(col)
                var["type"] = config.get("type", "nominal")
                var["cardinality"] = config.get("cardinality")
                var["encoding"] = config.get("mapping") # Name -> Int
                # Generate labels list sorted by int value
                if config.get("mapping"):
                    sorted_items = sorted(config["mapping"].items(), key=lambda x: x[1])
                    var["labels"] = [k for k, v in sorted_items]

        # 5. Target Variable
        target_info = self.metadata.get("target_variable")
        if isinstance(target_info, dict):
            name = target_info.get("name")
            if name:
                var = get_var(name)
                var["type"] = target_info.get("type")
                var["cardinality"] = target_info.get("cardinality")
                var["encoding"] = target_info.get("mapping")
                var["role"] = "target"
                if target_info.get("mapping"):
                    sorted_items = sorted(target_info["mapping"].items(), key=lambda x: x[1])
                    var["labels"] = [k for k, v in sorted_items]

        return new_meta
    
    def _generate_distribution_report(self, df, filepath):
        print(f"Generating intuitive distribution report to {filepath}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== Distribución del Dataset (Discretizado/Etiquetado) ===\n")
            f.write(f"Total registros: {len(df)}\n\n")
            
            for col in df.columns:
                f.write(f"--- Distribución de: {col} ---\n")
                counts = df[col].value_counts()
                percents = df[col].value_counts(normalize=True) * 100
                stats_df = pd.DataFrame({'Count': counts, 'Percentage': percents})
                f.write(stats_df.to_string())
                f.write("\n\n")
    
    def _apply_filters(self, df):
        """Aplica filtros de filas y eliminación de columnas definidos en metadata"""
        # Aplicar filtros de filas
        if 'row_filters' in self.metadata:
            for filter_rule in self.metadata['row_filters']:
                col = filter_rule['column']
                val = filter_rule['value']
                op = filter_rule.get('operator', '==')
                
                if col in df.columns:
                    print(f"Applying filter: {col} {op} {val}")
                    initial_rows = len(df)
                    if op == '==':
                        df = df[df[col] == val]
                    elif op == '!=':
                        df = df[df[col] != val]
                    elif op == '>':
                        df = df[df[col] > val]
                    elif op == '<':
                        df = df[df[col] < val]
                    elif op == '>=':
                        df = df[df[col] >= val]
                    elif op == '<=':
                        df = df[df[col] <= val]
                    print(f"Rows filtered: {initial_rows} -> {len(df)}")
        
        # Eliminar columnas
        if 'drop_columns' in self.metadata:
            cols_to_drop = [c for c in self.metadata['drop_columns'] if c in df.columns]
            if cols_to_drop:
                print(f"Dropping columns: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)
                
        return df

    def _rediscretize_variables(self, df):
        """Aplica rediscretización (agrupamiento) de variables según metadata"""
        # Buscar configuraciones de rediscretización en la sección 'rediscretization'
        if 'rediscretization' in self.metadata:
            for col, config in self.metadata['rediscretization'].items():
                if col in df.columns:
                    if 'mapping' in config and 'grouping_strategy' in config:
                        print(f"Rediscretizing column: {col} using strategy: {config['grouping_strategy']}")
                        
                        # Invertir el mapeo: de {Grupo: [val1, val2]} a {val1: Grupo, val2: Grupo}
                        value_to_group = {}
                        for group, values in config['mapping'].items():
                            for val in values:
                                value_to_group[val] = group
                        
                        # Aplicar mapeo
                        # Usar map y llenar nulos con una categoría 'Other' o mantener original si se desea
                        # Aquí asumimos que todo lo que no está en el mapa se va a 'Other' si no se encuentra
                        df[col] = df[col].map(value_to_group).fillna('Other')
                        
                        # Actualizar metadata para categorical_encoding si no existe
                        if 'categorical_encoding' not in self.metadata:
                            self.metadata['categorical_encoding'] = {}
                        
                        # Crear mapping para la nueva variable agrupada
                        unique_groups = sorted(list(config['mapping'].keys()) + ['Other'])
                        # Filtrar 'Other' si no se usó
                        if 'Other' not in df[col].unique():
                            unique_groups.remove('Other')
                            
                        self.metadata['categorical_encoding'][col] = {
                            'type': 'nominal',
                            'mapping': {val: i for i, val in enumerate(unique_groups)},
                            'cardinality': len(unique_groups)
                        }
        return df

    def _discretize_continuous(self, df, to_labels=False):
        for col, config in self.metadata['continuous_to_categorical'].items():
            if col in df.columns:
                # Si to_labels es True, usamos los labels definidos en metadata
                # Si es False, usamos range(cardinality) para obtener enteros
                labels = config['labels'] if to_labels else range(config['cardinality'])
                
                df[col] = pd.cut(
                    df[col], 
                    bins=config['bins'] + [np.inf],
                    labels=labels,
                    right=config['right']
                )
                # Convertir a string si usamos labels para evitar problemas de tipo category
                if to_labels:
                    df[col] = df[col].astype(str)
        return df
    
    def _decode_one_hot(self, df, to_labels=False):
        for col, config in self.metadata['one_hot_to_decode'].items():
            # Encontrar qué columna one-hot está activa
            one_hot_cols = [c for c in config['columns'] if c in df.columns]
            if one_hot_cols:
                # Mapeo de nombre de columna a índice o etiqueta
                if to_labels:
                    # Usar el mapping definido en metadata: "0": "African American"
                    # Pero idxmax devuelve el nombre de la columna (ej. race:AfricanAmerican)
                    # Necesitamos mapear nombre de columna -> etiqueta limpia
                    
                    # Primero obtenemos el índice (0, 1, 2...) basado en el orden de columns
                    col_to_idx = {c: str(i) for i, c in enumerate(config['columns'])}
                    
                    # Luego mapeamos índice -> etiqueta
                    idx_to_label = config['mapping']
                    
                    # Combinamos: columna -> etiqueta
                    col_to_label = {c: idx_to_label[col_to_idx[c]] for c in one_hot_cols if col_to_idx[c] in idx_to_label}
                    
                    df[col] = df[one_hot_cols].idxmax(axis=1).map(col_to_label)
                else:
                    df[col] = df[one_hot_cols].idxmax(axis=1).map(
                        {c: i for i, c in enumerate(one_hot_cols)}
                    )
                
                df = df.drop(columns=one_hot_cols)
        return df
    
    def _encode_categorical(self, df):
        for col, config in self.metadata['categorical_encoding'].items():
            if col in df.columns:
                # Si la columna ya es numérica (porque no se llamó con to_labels=True antes o ya se procesó), saltar
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue
                    
                # Si es string (etiquetas), mapear a enteros
                print(f"Encoding categorical: {col}")
                df[col] = df[col].map(config['mapping'])
                
                # Manejar valores que no se mapearon (NaN)
                if df[col].isnull().any():
                    print(f"Warning: NaN values found in {col} after encoding. Filling with -1.")
                    df[col] = df[col].fillna(-1).astype(int)
        
        # También necesitamos codificar las variables continuas que fueron discretizadas a etiquetas
        for col, config in self.metadata['continuous_to_categorical'].items():
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Encoding discretized: {col}")
                # Crear mapa inverso de labels -> enteros
                label_map = {label: i for i, label in enumerate(config['labels'])}
                df[col] = df[col].map(label_map)
                
        return df

    def _auto_encode_remaining(self, df):
        """Codifica automáticamente cualquier columna de texto restante"""
        object_cols = df.select_dtypes(include=['object']).columns
        
        if 'categorical_encoding' not in self.metadata:
            self.metadata['categorical_encoding'] = {}
            
        for col in object_cols:
            print(f"Auto-encoding column: {col}")
            codes, uniques = pd.factorize(df[col])
            df[col] = codes
            
            # Actualizar metadata en memoria para cardinalidades
            self.metadata['categorical_encoding'][col] = {
                'type': 'nominal',
                'mapping': {str(val): int(code) for code, val in enumerate(uniques)},
                'cardinality': len(uniques)
            }
        return df
    
    def get_cardinalities(self):
        """Retorna las cardinalidades de todas las variables"""
        cardinalities = {}
        
        for col, config in self.metadata['continuous_to_categorical'].items():
            cardinalities[col] = config['cardinality']
        
        for col, config in self.metadata['one_hot_to_decode'].items():
            cardinalities[col] = config['cardinality']
        
        for col, config in self.metadata['categorical_encoding'].items():
            cardinalities[col] = config['cardinality']
        
        return cardinalities
    
    def decode_solution(self, encoded_values):
        """Decodifica una solución a valores legibles"""
        decoded = {}
        feature_order = self.metadata['feature_order']
        
        for i, col in enumerate(feature_order):
            value = encoded_values[i]
            
            # Buscar en qué tipo de variable está
            if col in self.metadata['continuous_to_categorical']:
                labels = self.metadata['continuous_to_categorical'][col]['labels']
                decoded[col] = labels[int(value)]
            
            elif col in self.metadata['categorical_encoding']:
                mapping = self.metadata['categorical_encoding'][col]['mapping']
                inv_mapping = {v: k for k, v in mapping.items()}
                decoded[col] = inv_mapping[int(value)]
            
            elif col in self.metadata['one_hot_to_decode']:
                mapping = self.metadata['one_hot_to_decode'][col]['mapping']
                decoded[col] = mapping[str(int(value))]
        
        return decoded
    

    # Uso:
if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

    # Check if raw directory exists
    if not os.path.exists(RAW_DIR):
        print(f"Error: Raw data directory not found at {RAW_DIR}")
        sys.exit(1)

    # Get list of dataset folders
    dataset_folders = [f for f in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, f))]

    if not dataset_folders:
        print(f"Error: No dataset folders found in {RAW_DIR}")
        sys.exit(1)

    # Create processed directory if it doesn't exist
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        print(f"Created directory: {PROCESSED_DIR}")

    for dataset_name in dataset_folders:
        print(f"Processing dataset: {dataset_name}")
        dataset_path = os.path.join(RAW_DIR, dataset_name)
        
        # Look for metadata and csv
        json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

        if not json_files or not csv_files:
            print(f"Skipping {dataset_name}: Missing JSON or CSV file.")
            continue
        
        # Use the first found files
        metadata_path = os.path.join(dataset_path, json_files[0])
        csv_path = os.path.join(dataset_path, csv_files[0])

        try:
            # Load data
            print(f"Loading data from {csv_path}")
            df_raw = pd.read_csv(csv_path)
            
            # Initialize preprocessor
            print(f"Loading metadata from {metadata_path}")
            preprocessor = DatasetPreprocessor(metadata_path)
            
            # Create output folder first to save stats
            output_folder = os.path.join(PROCESSED_DIR, dataset_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Apply filters manually first just to generate stats on the filtered data BEFORE encoding
            # This is a bit redundant with preprocess() but useful for the report
            df_stats = df_raw.copy()
            if 'row_filters' in preprocessor.metadata:
                for filter_rule in preprocessor.metadata['row_filters']:
                    col = filter_rule['column']
                    val = filter_rule['value']
                    op = filter_rule.get('operator', '==')
                    if col in df_stats.columns and op == '==': # Simple support for equality in stats preview
                         df_stats = df_stats[df_stats[col] == val]

            # Generate distribution statistics
            stats_path = os.path.join(output_folder, f"{dataset_name}_distribution.txt")
            print(f"Generating distribution statistics to {stats_path}...")
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Distribución del Dataset: {dataset_name} ===\n")
                f.write(f"Total registros (filtrados): {len(df_stats)}\n\n")
                
                for col in df_stats.columns:
                    f.write(f"--- Distribución de: {col} ---\n")
                    counts = df_stats[col].value_counts()
                    percents = df_stats[col].value_counts(normalize=True) * 100
                    stats_df = pd.DataFrame({'Count': counts, 'Percentage': percents})
                    f.write(stats_df.to_string())
                    f.write("\n\n")

            # Preprocess (this will apply filters again properly and encode)
            print("Preprocessing...")
            df_processed = preprocessor.preprocess(df_raw, output_folder=output_folder)
            
            # Save processed data
            output_path = os.path.join(output_folder, f"{dataset_name}_processed.csv")
            df_processed.to_csv(output_path, index=False)
            print(f"Saved processed data to: {output_path}")
            
            # Save codification metadata
            metadata_path = os.path.join(output_folder, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                # Convert numpy types to python types for JSON serialization
                def convert(o):
                    if isinstance(o, np.integer): return int(o)
                    if isinstance(o, np.floating): return float(o)
                    if isinstance(o, np.ndarray): return o.tolist()
                    return str(o)
                
                # Export metadata in the new variable-centric format
                final_metadata = preprocessor._export_metadata_structure()
                json.dump(final_metadata, f, indent=2, default=convert, ensure_ascii=False)
            print(f"Saved codification metadata to: {metadata_path}")
            
            cardinalities = preprocessor.get_cardinalities()
            print(f"Cardinalities: {cardinalities}")

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()