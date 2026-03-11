import pandas as pd
import numpy as np


def generate_training_data(filename='training_data.csv', n_samples=50, seed=41):

    #np.random.seed(seed)

    # 1. Bandas espectrales (reflectancias típicas Sentinel-2 en trigo)
    B02 = np.random.uniform(0.03, 0.20, n_samples)  # Azul
    B03 = np.random.uniform(0.05, 0.25, n_samples)  # Verde
    B04 = np.random.uniform(0.03, 0.20, n_samples)  # Rojo
    B08 = np.random.uniform(0.20, 0.60, n_samples)  # NIR

    # 2. Rendimiento objetivo (t/ha)
    rendimiento_objetivo = np.random.uniform(3.0, 9.0, n_samples)

    # 3. NDVI derivado de las bandas
    ndvi = (B08 - B04) / (B08 + B04 + 1e-8)

    # 4. Dosis de nitrógeno basada en lógica agronómica
    #    - A mayor NDVI, menor necesidad de N (el cultivo ya está sano)
    #    - A mayor rendimiento objetivo, mayor necesidad de N
    #    - La banda verde (B03) correlaciona con contenido de clorofila;
    #      alta clorofila → menos N necesario
    #    - La banda azul (B02) aporta info sobre estrés hídrico
    base_n      = 80.0
    coef_ndvi   = -120.0   # NDVI alto → menos N
    coef_rend   = 15.0     # Rendimiento alto → más N
    coef_b03    = -60.0    # Verde alto → menos N (más clorofila)
    coef_b02    = 40.0     # Azul alto → más N (posible estrés)

    noise = np.random.normal(0, 5.0, n_samples)

    nitrogen_dosis = (
        base_n
        + coef_ndvi  * ndvi
        + coef_rend  * rendimiento_objetivo
        + coef_b03   * B03
        + coef_b02   * B02
        + noise
    )

    # 5. Construir DataFrame y guardar
    df = pd.DataFrame({
        'B02': B02,
        'B03': B03,
        'B04': B04,
        'B08': B08,
        'rendimiento_objetivo': rendimiento_objetivo,
        'nitrogen_dosis': nitrogen_dosis,
    })

    df.to_csv(filename, index=False)
    print(f"Archivo '{filename}' con {n_samples} muestras generado correctamente.")
    return df


if __name__ == "__main__":
    generate_training_data()
