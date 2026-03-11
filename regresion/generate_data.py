import pandas as pd
import numpy as np

def generate_training_data(filename='ndvi_data.csv', n_samples=1000, seed=42):
    """
    Genera datos sintéticos para el entrenamiento del modelo NDVI.
    Basado en la relación: ndvi_peak ~ ndvi_app + n_coverage
    """
    np.random.seed(seed)
    
    # 1. Generar variables independientes
    data = {
        'ndvi_app': np.random.uniform(0.3, 0.7, n_samples),
        'n_coverage': np.random.uniform(0, 150, n_samples),
    }
    df = pd.DataFrame(data)
    
    # 2. Generar el target (ndvi_peak) con ruido normal
    # Coeficientes basados en lógica agronómica simplificada
    intercept = 0.2
    coef_ndvi_app = 0.6
    coef_n_coverage = 0.001
    noise = np.random.normal(0, 0.02, n_samples)
    
    df['ndvi_peak'] = intercept + (coef_ndvi_app * df['ndvi_app']) + (coef_n_coverage * df['n_coverage']) + noise
    
    # 3. Guardar en CSV
    df.to_csv(filename, index=False)
    print(f"Archivo '{filename}' con {n_samples} muestras generado correctamente.")

if __name__ == "__main__":
    generate_training_data()
