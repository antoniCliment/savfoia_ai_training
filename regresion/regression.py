import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class NDVIModel:
    """
    Clase para entrenar el modelo NDVI y calcular el nitrógeno necesario
    para alcanzar un pico de NDVI objetivo dado el estado actual del cultivo.
    """
    def __init__(self, model_path='modelo_ndvi_wheat.pkl'):
        self.model_path = model_path
        self.model = None

    def train(self, data_path='ndvi_data.csv'):
        """Entrena el modelo lineal con los datos proporcionados."""
        print("--- Entrenando el modelo ---")
        df = pd.read_csv(data_path)
        
        X = df[['ndvi_app', 'n_coverage']]
        y = df['ndvi_peak']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Evaluación
        y_pred = self.model.predict(X_test)
        print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        
        # Guardar
        joblib.dump(self.model, self.model_path)
        print(f"Modelo guardado en: {self.model_path}\n")
        return df

    def load(self):
        """Carga el modelo desde el disco."""
        self.model = joblib.load(self.model_path)
        return self.model

    def optimize_nitrogen(self, target_ndvi_peak, ndvi_app):
        """
        Calcula el nitrógeno de cobertera necesario para alcanzar un pico de NDVI objetivo.
        Acepta arrays de valores para el procesamiento por lotes.

        Entradas:
            target_ndvi_peak : array-like  -- Pico de NDVI objetivo para cada parcela
            ndvi_app         : array-like  -- Estado actual de NDVI en aplicación

        Devuelve:
            np.ndarray con el nitrógeno necesario (kg/ha) para cada entrada.

        Basado en la inversión de: ndvi_peak = b0 + b1*ndvi_app + b2*n_coverage
            --> n_coverage = (ndvi_peak - b0 - b1*ndvi_app) / b2
        """
        if self.model is None:
            self.load()

        intercept = self.model.intercept_
        b_ndvi = self.model.coef_[0]
        b_n = self.model.coef_[1]

        target_arr = np.atleast_1d(target_ndvi_peak)
        ndvi_app_arr = np.atleast_1d(ndvi_app)

        n_required = (target_arr - intercept - (b_ndvi * ndvi_app_arr)) / b_n
        return n_required

def visualize_results(model_wrapper, df, ndvi_app, target_peak, n_required):
    """Genera y guarda una visualización clara de la optimización, un panel por escenario."""
    print("--- Generando visualización ---")

    ndvi_apps   = np.atleast_1d(ndvi_app)
    target_peaks = np.atleast_1d(target_peak)
    n_requireds  = np.atleast_1d(n_required)
    n_scenarios  = len(ndvi_apps)

    # Paleta de colores clara y distinguible por escenario
    palette = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    # Un subplot por escenario, en fila
    fig, axes = plt.subplots(1, n_scenarios, figsize=(6 * n_scenarios, 6), sharey=True)
    if n_scenarios == 1:
        axes = [axes]

    fig.suptitle('Optimización de Nitrógeno por Parcela', fontsize=16, fontweight='bold', y=1.01)

    n_range = np.linspace(0, 300, 200)
    margin = 0.05

    for i, ax in enumerate(axes):
        current_app    = ndvi_apps[i]
        current_target = target_peaks[i]
        current_n      = n_requireds[i]
        color          = palette[i % len(palette)]

        # 1. Datos históricos relevantes para este NDVI_app
        mask = (df['ndvi_app'] >= current_app - margin) & (df['ndvi_app'] <= current_app + margin)
        df_sub = df[mask]
        ax.scatter(df_sub['n_coverage'], df_sub['ndvi_peak'],
                   color='#90CAF9', alpha=0.4, s=18, label='Datos históricos', zorder=1)

        # 2. Curva de predicción del modelo para este NDVI_app
        X_viz = pd.DataFrame({'ndvi_app': [current_app] * len(n_range), 'n_coverage': n_range})
        y_pred = model_wrapper.model.predict(X_viz)
        ax.plot(n_range, y_pred, color=color, linewidth=2.5, label='Curva del modelo', zorder=2)

        # 3. Líneas guía al punto óptimo
        ax.axhline(y=current_target, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.axvline(x=current_n,      color='gray', linestyle='--', linewidth=1, alpha=0.6)

        # 4. Punto óptimo destacado
        ax.scatter([current_n], [current_target], color=color, s=160,
                   edgecolor='black', linewidth=1.5, zorder=5,
                   label=f'N necesario: {current_n:.1f} kg/ha')

        # 5. Anotación del valor
        ax.annotate(f'{current_n:.1f} kg/ha',
                    xy=(current_n, current_target),
                    xytext=(current_n + 8, current_target - 0.025),
                    fontsize=10, fontweight='bold', color=color)

        # Títulos y etiquetas
        ax.set_title(f'Parcela {i+1}\nNDVI_app={current_app:.2f}  →  NDVI_peak={current_target:.2f}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Nitrógeno de Cobertera (kg/ha)', fontsize=10)
        if i == 0:
            ax.set_ylabel('Pico de NDVI Estimado', fontsize=10)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    output_file = 'optimizacion_nitrogeno_lote.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualización guardada como '{output_file}'.\n")
    plt.show()

def main():
    # 1. Entrenamiento
    ndvi_sys = NDVIModel()
    data_df = ndvi_sys.train()

    # 2. Inferencia por Lotes:
    #    Dado el NDVI actual (ndvi_app) y el pico de NDVI objetivo (target),
    #    calculamos el nitrógeno de cobertera necesario para cada parcela.
    ndvi_apps  = [0.4, 0.5, 0.6]
    ndvi_peaks = [0.6, 0.7, 0.8]

    n_necesarios = ndvi_sys.optimize_nitrogen(ndvi_peaks, ndvi_apps)

    print("--- Inferencia por Lotes: N necesario por parcela ---")
    for app, peak, n_req in zip(ndvi_apps, ndvi_peaks, n_necesarios):
        print(f"  NDVI_app={app:.2f}, NDVI_peak objetivo={peak:.2f} --> N necesario: {n_req:.2f} kg/ha")
    print()

    # 3. Visualización de Resultados
    visualize_results(ndvi_sys, data_df, ndvi_apps, ndvi_peaks, n_necesarios)

if __name__ == "__main__":
    main()