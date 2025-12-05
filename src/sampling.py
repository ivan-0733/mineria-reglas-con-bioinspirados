import pandas as pd
import json
import os
import sys

def create_multilevel_sample(df, sample_size=5000, random_state=42):
    """
    Muestreo en dos niveles para mejor representatividad
    """
    samples = []
    
    # Nivel 1: Estratificar por diabetes
    for diabetes_val in df['diabetes'].unique():
        df_stratum = df[df['diabetes'] == diabetes_val]
        n_samples = int(sample_size * len(df_stratum) / len(df))
        
        # Nivel 2: Dentro de cada estrato de diabetes, estratificar por age
        for age_val in df_stratum['age'].unique():
            df_sub = df_stratum[df_stratum['age'] == age_val]
            n_sub = int(n_samples * len(df_sub) / len(df_stratum))
            
            if n_sub > 0 and len(df_sub) >= n_sub:
                sample = df_sub.sample(n=n_sub, random_state=random_state)
                samples.append(sample)
    
    sample_df = pd.concat(samples, ignore_index=True)
    
    # Ajuste final si es necesario
    if len(sample_df) < sample_size:
        remaining = sample_size - len(sample_df)
        additional = df[~df.index.isin(sample_df.index)].sample(
            n=remaining, random_state=random_state
        )
        sample_df = pd.concat([sample_df, additional], ignore_index=True)
    
    return sample_df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def validate_sample(original_df, sample_df, output_path=None):
    """
    Compara distribuciones entre dataset original y muestra
    """
    output = []
    output.append("=== Validación de Muestra ===\n")
    
    categorical_cols = ['gender', 'age', 'location', 'hypertension', 
                       'heart_disease', 'smoking_history', 'bmi', 
                       'hbA1c_level', 'blood_glucose_level', 'diabetes', 'race']
    
    # Filter columns that actually exist in the dataframe
    categorical_cols = [col for col in categorical_cols if col in original_df.columns]
    
    for col in categorical_cols:
        output.append(f"--- {col} ---")
        orig_dist = original_df[col].value_counts(normalize=True).sort_index()
        samp_dist = sample_df[col].value_counts(normalize=True).sort_index()
        
        comparison = pd.DataFrame({
            'Original %': orig_dist * 100,
            'Muestra %': samp_dist * 100,
            'Diferencia': (samp_dist - orig_dist) * 100
        })
        output.append(comparison.round(2).to_string())
        output.append("\n")
    
    output_text = "\n".join(output)
    print(output_text)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"✅ Reporte de validación guardado en: {output_path}")
    

# Uso:
if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    
    DATASET_NAME = 'diabetes_dataset'
    INPUT_DIR = os.path.join(PROCESSED_DIR, DATASET_NAME)
    SAMPLE_DIR = os.path.join(PROCESSED_DIR, 'sample')
    
    CSV_PATH = os.path.join(INPUT_DIR, f'{DATASET_NAME}_processed.csv')
    # Try to find metadata file
    METADATA_PATH = os.path.join(INPUT_DIR, 'metadata.json')
    if not os.path.exists(METADATA_PATH):
        METADATA_PATH = os.path.join(INPUT_DIR, 'codification.json')

    # Cargar dataset original y metadata
    if not os.path.exists(CSV_PATH):
        print(f"Error: No se encuentra el archivo {CSV_PATH}")
        sys.exit(1)
        
    print(f"Cargando datos de: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    if os.path.exists(METADATA_PATH):
        print(f"Cargando metadata de: {METADATA_PATH}")
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        print("Warning: Metadata not found.")
        metadata = {}

    # 1. Crear muestra estratificada
    print("Creando muestra estratificada...")
    sample_df = create_multilevel_sample(df, sample_size=5000)

    # 3. Guardar muestra
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    SAMPLE_CSV_PATH = os.path.join(SAMPLE_DIR, 'diabetes_sample_5k.csv')
    sample_df.to_csv(SAMPLE_CSV_PATH, index=False)
    print(f"✅ Muestra guardada en: {SAMPLE_CSV_PATH}")

    # 2. Validar distribuciones y guardar reporte
    VALIDATION_REPORT_PATH = os.path.join(SAMPLE_DIR, 'validation_report.txt')
    validate_sample(df, sample_df, output_path=VALIDATION_REPORT_PATH)

    # 4. Guardar metadata actualizado
    SAMPLE_METADATA_PATH = os.path.join(SAMPLE_DIR, 'sample_metadata.json')
    
    metadata_sample = metadata.copy()
    metadata_sample['dataset_info'] = {
        "original_size": len(df),
        "sample_size": len(sample_df),
        "sampling_method": "stratified",
        "stratification_vars": ["diabetes", "age", "location"],
        "random_state": 42
    }

    with open(SAMPLE_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata_sample, f, indent=2, ensure_ascii=False)
    print(f"✅ Metadata de muestra guardada en: {SAMPLE_METADATA_PATH}")

    print(f"✅ Registros finales: {len(sample_df)}")
    print(f"✅ Casos diabetes positivos: {sample_df['diabetes'].sum()}")