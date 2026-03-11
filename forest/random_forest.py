"""
random_forest.py: Predictor de dosis de nitrógeno con Random Forest
Inputs: B02, B03, B04, B08 (bandas Sentinel-2) y rendimiento_objetivo.
Output: nitrogen_dosis (kg/ha).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from generate_data import generate_training_data


# Configuración del dominio
FEATURES     = ['B02', 'B03', 'B04', 'B08', 'rendimiento_objetivo']
TARGET       = 'nitrogen_dosis'

class NitrogenRFModel:
    """Clase para manejar el entrenamiento e inferencia del Random Forest."""

    def __init__(self, model_path='modelo_nitrogeno_rf.pkl'):
        self.model_path = model_path
        self.model: RandomForestRegressor | None = None

    # Entrenamiento
    def train(self, data_path='training_data.csv'):
        """Entrena el modelo Random Forest y lo guarda en disco."""
        print("=" * 60)
        print("  ENTRENAMIENTO DEL MODELO RANDOM FOREST")
        print("=" * 60)

        df = pd.read_csv(data_path)
        X  = df[FEATURES]
        y  = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,         # árboles completos
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        # Métricas de evaluación
        y_pred = self.model.predict(X_test)
        r2     = r2_score(y_test, y_pred)
        mae    = mean_absolute_error(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\n  Muestras entrenamiento : {len(X_train)}")
        print(f"  Muestras prueba        : {len(X_test)}")
        print(f"  R²  Score              : {r2:.4f}")
        print(f"  MAE                    : {mae:.2f} kg/ha")
        print(f"  RMSE                   : {rmse:.2f} kg/ha")

        # Importancia de características
        fi = self.model.feature_importances_
        fi_dict = dict(zip(FEATURES, fi))
        print("\n  Importancia de características:")
        for feat, imp in sorted(fi_dict.items(), key=lambda x: -x[1]):
            bar = '#' * int(imp * 40)
            print(f"    {feat:<25s} {imp:.4f}  {bar}")

        # Guardar
        joblib.dump(self.model, self.model_path)
        print(f"\n  Modelo guardado en: {self.model_path}\n")
        return df

    # Carga
    def load(self):
        """Carga el modelo desde disco."""
        self.model = joblib.load(self.model_path)
        return self.model

    # Predicción directa
    def predict(self, B02, B03, B04, B08, rendimiento_objetivo):
        """
        Predice la dosis de nitrógeno (kg/ha) para un conjunto de parcelas.
        Todos los parámetros pueden ser escalares o arrays de igual longitud.
        """
        if self.model is None:
            self.load()

        df_input = pd.DataFrame({
            'B02': np.atleast_1d(B02),'B03': np.atleast_1d(B03),
            'B04': np.atleast_1d(B04),'B08': np.atleast_1d(B08),
            'rendimiento_objetivo': np.atleast_1d(rendimiento_objetivo),
        })
        return self.model.predict(df_input)


def main():
    # 1. Generar datos (si no existen, se crean)
    import os
    data_path = 'training_data.csv'
    if not os.path.exists(data_path):
        generate_training_data(filename=data_path)

    # 2. Entrenar modelo
    rf_model = NitrogenRFModel()
    data_df  = rf_model.train(data_path=data_path)

    # 3. Definir parcelas de ejemplo
    #    Cada parcela tiene sus bandas espectrales y un rendimiento objetivo
    parcelas = [
        {'etiqueta':'Parcela A  (bajo NDVI)','B02': 0.10,'B03': 0.12,'B04': 0.15,'B08': 0.25,'rendimiento_objetivo': 5.0,},
        {'etiqueta':'Parcela B  (NDVI medio)','B02': 0.07,'B03': 0.10,'B04': 0.08,'B08': 0.40,'rendimiento_objetivo': 6.5,},
        {'etiqueta':'Parcela C  (alto NDVI)','B02': 0.05,'B03': 0.08,'B04': 0.05,'B08': 0.55,'rendimiento_objetivo': 8.0,},
    ]

    # Inferencia
    for p in parcelas:
        ndvi = (p['B08'] - p['B04']) / (p['B08'] + p['B04'] + 1e-8)
        dosis = rf_model.predict(
            p['B02'], p['B03'], p['B04'], p['B08'], p['rendimiento_objetivo']
        )[0]
        print(
            f"  {p['etiqueta']:<30s}  "
            f"NDVI={ndvi:.2f}  "
            f"Rend.={p['rendimiento_objetivo']:.1f} t/ha  "
            f"Dosis N: {dosis:.1f} kg/ha"
        )
    print()


if __name__ == "__main__":
    main()