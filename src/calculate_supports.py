import pandas as pd
import json
import os
import sys

def calculate_supports(df):
    """
    Calcula los soportes (frecuencias relativas) de cada valor
    para todas las columnas del dataframe.
    Retorna un diccionario con la estructura.
    """
    total_rows = len(df)
    supports = {
        "meta": {
            "total_rows": total_rows,
            "generated_at": pd.Timestamp.now().isoformat()
        },
        "variables": {}
    }
    
    for col in df.columns:
        # Value counts normalize=True da la frecuencia relativa (soporte)
        val_counts = df[col].value_counts(normalize=True)
        # Convertir a diccionario con claves string para JSON
        supports["variables"][col] = {
            str(k): float(v) for k, v in val_counts.items()
        }
    
    return supports

def process_file(csv_path, output_path):
    """Lee un CSV, calcula soportes y guarda el JSON."""
    print(f"Procesando: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        supports = calculate_supports(df)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(supports, f, indent=2, ensure_ascii=False)
        print(f"✅ Soportes guardados en: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error procesando {csv_path}: {e}")
        return False

def main():
    # Definir rutas base
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    
    # 1. Buscar y procesar datasets procesados normales
    # Estructura esperada: data/processed/<dataset_name>/<dataset_name>_processed.csv
    if os.path.exists(PROCESSED_DIR):
        for item in os.listdir(PROCESSED_DIR):
            item_path = os.path.join(PROCESSED_DIR, item)
            if os.path.isdir(item_path) and item != "sample":
                # Buscar archivo _processed.csv
                csv_files = [f for f in os.listdir(item_path) if f.endswith('_processed.csv')]
                if csv_files:
                    csv_path = os.path.join(item_path, csv_files[0])
                    output_path = os.path.join(item_path, 'supports.json')
                    process_file(csv_path, output_path)

    # 2. Procesar la muestra (sample)
    # Estructura esperada: data/processed/sample/diabetes_sample_5k.csv
    sample_dir = os.path.join(PROCESSED_DIR, 'sample')
    if os.path.exists(sample_dir):
        csv_files = [f for f in os.listdir(sample_dir) if f.endswith('.csv') and 'sample' in f]
        for csv_file in csv_files:
            csv_path = os.path.join(sample_dir, csv_file)
            output_path = os.path.join(sample_dir, 'sample_supports.json')
            process_file(csv_path, output_path)

if __name__ == "__main__":
    main()
